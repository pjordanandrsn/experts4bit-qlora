### Rental drivers report unknown probes and check token scope before staging

Heartbeats now report consecutive unknown probes without using that count to stop
a lane. TC1, P127 and FAM check token scope on the controller before copying it;
read-only tokens pass, verified unsafe tokens refuse, and failed or unrecognized
verification continues without staging a token or using implicit authentication.
Locality keeps staging no token. Credentials and permission data are never logged.
