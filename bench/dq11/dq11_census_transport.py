"""Operational direct SSH staging and an external, absolute phase watchdog."""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path

STAGING_SECONDS = 180
RESERVE_SECONDS = 300


def record(path, **row):
    with Path(path).open('a') as out:
        out.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), **row)) + '\n')


def direct_available(offer):
    try:
        return (ipaddress.ip_address(offer['public_ipaddr']).version == 4
                and ipaddress.ip_address(offer['public_ipaddr']).is_global
                and type(offer['direct_port_count']) is int and offer['direct_port_count'] > 0)
    except (KeyError, ValueError, TypeError):
        return False


def direct_endpoint(instance, expected_id):
    if str(instance.get('id')) != str(expected_id):
        raise ValueError('foreign instance record')
    host = str(instance.get('public_ipaddr', ''))
    if ipaddress.ip_address(host).version != 4 or not ipaddress.ip_address(host).is_global:
        raise ValueError('direct SSH needs a public IP')
    bindings = instance.get('ports', {}).get('22/tcp', [])
    if not isinstance(bindings, list) or len(bindings) != 1:
        raise ValueError('direct SSH port mapping absent or ambiguous')
    port = str(bindings[0].get('HostPort', ''))
    if not port.isdigit() or not 1 <= int(port) <= 65535:
        raise ValueError('invalid direct SSH port')
    return host, int(port)


@contextmanager
def process_group(command):
    """Keep the leader unreaped until its group is signalled, preventing PGID reuse."""
    if not command:
        raise ValueError('empty watchdog command')
    child = subprocess.Popen(command, start_new_session=True)
    previous = {}

    def terminate(signum, frame):
        raise InterruptedError('watchdog interrupted')

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, terminate)
        yield child
    finally:
        # No poll()/wait() before this signal: even an exited leader remains a
        # zombie with its PID reserved until wait() below. WNOWAIT observes only.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            # Darwin reports EPERM for a group containing only an unreaped
            # zombie; Linux reports ESRCH. Never suppress it for a live leader.
            if sys.platform != 'darwin' or not exited_unreaped(child):
                raise
            try:
                os.getpgid(child.pid)
            except ProcessLookupError:
                pass
            else:
                raise
        child.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def exited_unreaped(child):
    status = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT | os.WNOHANG)
    return status is not None and status.si_pid == child.pid


def supervised(command, seconds):
    """The child cannot cancel this parent's monotonic timeout with SIGALRM."""
    if seconds <= 0:
        raise TimeoutError('absolute work deadline exhausted')
    with process_group(command) as child:
        deadline = time.monotonic() + seconds
        while not exited_unreaped(child):
            left = deadline - time.monotonic()
            if left <= 0:
                raise subprocess.TimeoutExpired(command, seconds)
            time.sleep(min(.02, left))
    return child.returncode


@contextmanager
def direct_connection(ssh, scp, log, budget):
    """Authenticate once; SCP reuses this master so its rate excludes handshake."""
    if not ssh:
        raise ValueError('empty direct SSH command')
    with tempfile.TemporaryDirectory(prefix='dq11-ssh-') as directory:
        socket = str(Path(directory) / 'control')
        command = [*ssh[:-1], '-M', '-S', socket, '-o', 'ControlPersist=no', '-N', ssh[-1]]
        start = time.monotonic()
        with process_group(command) as master:
            while not Path(socket).exists():
                if exited_unreaped(master):
                    raise ValueError('direct SSH setup failed')
                if time.monotonic() - start >= min(30, budget):
                    raise TimeoutError('direct SSH setup deadline exhausted')
                time.sleep(.02)
            setup = time.monotonic() - start
            record(log, phase='connection-setup', elapsed_seconds=setup)
            yield [*scp, '-o', f'ControlPath={socket}', '-o', 'ControlMaster=no',
                   '-o', 'ProxyCommand=false'], setup


def projected_seconds(total_bytes, first_bytes, elapsed):
    if first_bytes <= 0 or elapsed <= 0:
        raise ValueError('invalid staging rate sample')
    return elapsed * total_bytes / first_bytes


def staging_time_left():
    try:
        return int(os.environ['E4B_RENT_DEADLINE_EPOCH']) - time.time() - RESERVE_SECONDS
    except (KeyError, ValueError) as exc:
        raise ValueError('missing or invalid absolute staging deadline') from exc


def stage(files, scp, remote, log, *, ssh=None, clock=time.monotonic, run=supervised):
    ordered = sorted((Path(p) for p in files), key=lambda p: p.stat().st_size, reverse=True)
    if not ordered or len({p.name for p in ordered}) != len(ordered):
        raise ValueError('empty or colliding transport closure')
    total = sum(p.stat().st_size for p in ordered)
    first = ordered[0]
    # The entire stage and first bulk file share one deadline, never a retry clock.
    left = staging_time_left()
    budget = min(STAGING_SECONDS, left)
    if budget <= 0:
        raise TimeoutError('absolute staging deadline exhausted')
    started = clock()
    record(log, phase='stage-start', total_bytes=total, first_bytes=first.stat().st_size,
           first_file=first.name, budget_seconds=budget)
    try:
        connection = direct_connection(ssh, scp, log, budget) if ssh else nullcontext((scp, 0.0))
        with connection as (connected_scp, setup):
            transfer_start = clock()
            remaining = budget - (transfer_start - started)
            rc = run([*connected_scp, str(first), remote], remaining * first.stat().st_size / total)
            elapsed = clock() - transfer_start
            estimate = setup + projected_seconds(total, first.stat().st_size, elapsed)
            record(log, phase='first-bulk', returncode=rc, transfer_seconds=elapsed,
                   connection_setup_seconds=setup, projected_total_seconds=estimate)
            if rc or estimate > budget:
                raise ValueError('direct staging projected time exceeds budget')
            if len(ordered) > 1:
                rc = run([*connected_scp, *(str(p) for p in ordered[1:]), remote],
                         budget - (clock() - started))
                if rc:
                    raise ValueError('direct staging remainder failed')
            record(log, phase='stage-pass', elapsed_seconds=clock() - started, bytes=total)
    except (subprocess.TimeoutExpired, TimeoutError, ValueError, InterruptedError) as exc:
        record(log, phase='stage-refused', elapsed_seconds=clock() - started, error=str(exc))
        raise


def drive(command):
    # Instance GET only. No search, create, quote or controller policy changes here.
    sys.path.insert(0, os.environ['ADERTHA_REPO'])
    from adertha.compute.vast_provider import UrllibTransport, VastProvider, load_api_key

    class ReadOnly:
        def __init__(self):
            self.delegate = UrllibTransport(load_api_key())

        def request(self, method, path, **kwargs):
            if method != 'GET' or not path.startswith('/v0/instances/'):
                raise ValueError('direct endpoint resolver is instance-GET-only')
            return self.delegate.request(method, path, **kwargs)

    iid = os.environ['E4B_RENT_INSTANCE_ID']
    provider = VastProvider(ReadOnly(), run_label=os.environ['E4B_RENT_RUN_ID'])
    instance = provider.instance(iid)
    host, port = direct_endpoint(instance, iid)
    root = Path(os.environ['E4B_RENT_RUN_DIR'])
    known = root / 'direct-known_hosts'
    # Record-on-first-connect, matching the existing controller trust policy.
    options = ['-o', f'UserKnownHostsFile={known}', '-o', 'GlobalKnownHostsFile=/dev/null',
               '-o', 'StrictHostKeyChecking=accept-new', '-o', 'HashKnownHosts=no',
               '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=10',
               '-o', 'ServerAliveCountMax=2']
    record(root / 'direct-transport.jsonl', phase='endpoint', instance_id=iid,
           host=host, port=port, source='authenticated Vast instance public_ipaddr + ports[22/tcp]',
           host_key_policy='record-on-first-connect; refuse subsequent key changes',
           proxy_fallback=False)
    env = dict(os.environ, E4B_RENT_SSH_HOST=host, E4B_RENT_SSH_PORT=str(port),
               E4B_RENT_SSH=f'{host}:{port}', E4B_RENT_SSH_KNOWN_HOSTS=str(known),
               E4B_RENT_SSH_OPTS=shlex.join(options))
    os.execvpe(command[0], command, env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    d = sub.add_parser('drive'); d.add_argument('command', nargs=argparse.REMAINDER)
    p = sub.add_parser('phase'); p.add_argument('--seconds', type=float, required=True)
    p.add_argument('--deadline', type=int, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    s = sub.add_parser('stage'); s.add_argument('--scp', required=True)
    s.add_argument('--ssh', required=True)
    s.add_argument('--remote', required=True); s.add_argument('files', nargs='+')
    args = parser.parse_args()
    if args.mode in ('drive', 'phase') and not args.command:
        raise ValueError('empty watchdog command')
    if args.mode == 'drive':
        drive(args.command)
    elif args.mode == 'phase':
        cap = min(args.seconds, args.deadline - time.time() - RESERVE_SECONDS)
        log = Path('receipts/phase-timings.jsonl')
        start = time.monotonic()
        record(log, phase='start', name=args.name, command=args.command[0], cap_seconds=cap)
        result = 'refused'
        try:
            result = supervised(args.command, cap)
        finally:
            record(log, phase='end', name=args.name, command=args.command[0], result=result,
                   elapsed_seconds=time.monotonic() - start)
        sys.exit(result)
    else:
        staging_time_left()
        root = Path(os.environ['E4B_RENT_RUN_DIR'])
        known = os.environ['E4B_RENT_SSH_KNOWN_HOSTS']
        try:
            keys = subprocess.check_output(['ssh-keygen', '-lf', known], text=True,
                                           stderr=subprocess.DEVNULL).strip()
            if not keys:
                raise ValueError('no pinned host key')
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError('no pinned host key') from exc
        record(root / 'direct-transport.jsonl', phase='pinned-host-key', fingerprints=keys)
        stage(args.files, shlex.split(args.scp), args.remote, root / 'direct-transport.jsonl',
              ssh=shlex.split(args.ssh))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.TimeoutExpired, InterruptedError) as error:
        print(f'CENSUS_REFUSAL: {error}', file=sys.stderr)
        sys.exit(11)
