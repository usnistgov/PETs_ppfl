## Flower/Ray Client, Model, and Logging Behavior

This codebase currently follows the standard Flower simulation pattern:

- One data partition corresponds to one Flower client.
- One Flower client trains one local model update per federated round.

### Parameters that Affect How Many Clients are Trained

- `num_partitions` controls how many simulated client partitions are available when random partitioning is used.
- `num_clients` determines how many clients are trained when no data partition file is provided.
- When `data_partitions_file` is provided, the code reads the number of `num_client` entries in that file and uses those as the available clients and client data partitions.

### Parameters that Affect Concurrency

- `num_cpus` does not directly control how many models are trained.
- In Ray, `num_cpus` is a per-client resource reservation.
- Lowering `num_cpus` can allow more Flower clients to run at the same time, so logs from multiple clients may appear interleaved.
- Increasing `num_cpus` can force more sequential execution by making each client reserve more of the machine.