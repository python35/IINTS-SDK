# Pico Serial Protocol

Allowed commands: `PING`, `STATUS`, `LOCKOUT`.

Blocked commands: `DOSE`, `BOLUS`, `BASAL`, `PRIME`.

Expected safety response includes `hardware_actuation_enabled=false` and `locked=true`.
