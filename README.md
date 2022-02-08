# <img src="https://github.com/UofT-EcoSystem/DietCode/raw/main/figures/DietCode_text.png" height="48">

:point_up: *Please refer to the top-left corner of the README for the table of
contents*.

We summarize the goal and list the key source code under each specific key
component of <img
src="https://github.com/UofT-EcoSystem/DietCode/raw/main/figures/DietCode_text_black.png"
height="16">. Upon viewing each file, please search for the `<DietCode>` tag for
the detailed documentation.

#### Local Padding

- **Goal**: Mitigate the performance overhead brought by `if` predicates.
- **Source Code**:
  - [`src/te/schedule/message_passing.cc`](./src/te/schedule/message_passing.cc)
  - [`src/tir/transforms/vectorize_loop.cc`](./src/tir/transforms/vectorize_loop.cc)
  - [`src/te/operation/compute_op.cc`](./src/te/operation/compute_op.cc)

#### Loop Partitioning

- **Goal**: Mitigate the performance overhead brought by `if` predicates.
- **Source Code**:
  - [`src/tir/transforms/loop_partition.cc`](./src/tir/transforms/loop_partition.cc)
  - [`src/tir/transforms/thread_storage_sync.cc`](./src/tir/transforms/thread_storage_sync.cc)
