### Rental drivers detect dead processes and rebooted hosts

TC1, P127, FAM and locality drivers now share a probe of the launched PID and
its Linux start time, boot identity and uptime. It avoids self-matching process
counts and detects PID reuse, zombies and reboots. Two definite missing polls
end the wait; failed or malformed probes remain unknown and never stop a lane.
The shared controller helper is sent over SSH stdin, separate from staged pins.
