## Flower/Ray client, model, and logging behavior

This codebase currently follows the standard Flower simulation pattern:

- One data partition corresponds to one Flower client.
- One Flower client trains one local model update per federated round.

### Parameters that affect how many clients train

- `num_partitions` controls how many simulated client partitions are available when random partitioning is used.
- `num_clients` determines how many clients are trained when no data partition file is provided.
- When `data_partitions_file` is provided, the code reads the number of `num_client` entries in that file and uses those as the available clients and client data partitions.

### Parameters that affect concurrency

- `num_cpus` does not directly control how many models are trained.
- In Ray, `num_cpus` is a per-client resource reservation.
- Lowering `num_cpus` can allow more Flower clients to run at the same time, so logs from multiple clients may appear interleaved.
- Increasing `num_cpus` can force more sequential execution by making each client reserve more of the machine.

### Interpreting output

Current output may look like:

```
Client 0 | Epoch 1/100 | ...
Client 3 | Epoch 1/100 | ...
Client 1 | Epoch 1/100 | ...
```

It means multiple client processes are training concurrently, and Ray prints logs as each process emits them. The order is based on scheduling/runtime progress, not client id order.

### Planned cleanup

To reduce confusion in a future non-hotfix change:

- Document the relationship between partition files, Flower clients, and federated rounds directly in the configuration docs.
- Keep memory-related changes separate from naming/logging cleanup to avoid expanding the current hotfix scope.