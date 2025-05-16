# Fully-distributed ML framework implementation

- [`create_topologies`](https://github.com/msakarvadia/distributed_ml/tree/main/src/create_topo): directory for storing topology creation scripts
- [`experiments`](https://github.com/msakarvadia/distributed_ml/tree/main/src/experiments): directory for storing experiment + helper scripts
- [`data.py`](https://github.com/msakarvadia/distributed_ml/blob/main/src/data.py): contains utilities to partition data across workers and create OOD data
- [`decentralized_app.py`](https://github.com/msakarvadia/distributed_ml/blob/main/src/decentralized_app.py): experiment object (handles all logic related to configuring and running a single experiment)
- [`decentralized_client.py`](https://github.com/msakarvadia/distributed_ml/blob/main/src/decentralized_client.py): client object (handles all client specific utilities + aggregation utilities)
- [`modules.py`](https://github.com/msakarvadia/distributed_ml/blob/main/src/modules.py): defines datasets and their respecitive model architectures
- [`tasks.py`](https://github.com/msakarvadia/distributed_ml/blob/main/src/tasks.py): train + test utilities
- [`utils.py`](https://github.com/msakarvadia/distributed_ml/blob/main/src/utils.py): checkpointing + logging utilities
