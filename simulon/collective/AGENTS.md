# collective — CCL Decomposition

8 files. Decomposes abstract collectives (AllReduce, AllGather, ReduceScatter, AllToAll) into lists of P2PFlow records with explicit dependency chains.

## OVERVIEW

Independent of the DAG tracer. Given a collective type, group of GPU ranks, data size, and algorithm, produces P2PFlow records that the tracer converts into CommNodes. Ring topology from MockNcclGroup C++ binding with sequential fallback.

## STRUCTURE

collective/
├── init.py    # CCLDecomposer (Protocol), NCCLDecomposer, RCCLDecomposer (stub)
├── common.py      # P2PFlow dataclass — src, dst, bytes, flow_id, parent/child_flow_ids, conn_type
├── decompose.py   # decompose_collective() dispatcher + _REGISTRY map
├── ring.py        # Ring algorithms: reduce_scatter, all_gather, all_reduce, all_to_all
├── tree.py        # Tree AllReduce — binary tree via MockNcclGroup topology (reduce-up + broadcast-down)
├── collnet.py     # CollNet direct/chain (stubs — raises NotImplementedError)
├── nvls.py        # NVLS AllReduce (single-node star via NVSwitch) + NVLS Tree (multi-node: NVLS + inter-node tree)
└── calbusbw.py    # cal_busbw() — bandwidth calibration from NCCL profile, auto-selects best algorithm

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| New algorithm | New file + `decompose.py` | Implement function, register in `_REGISTRY` |
| New collective type | `ring.py` (or algo file) + `decompose.py` | Add to dispatcher |
| BW calibration / auto-select | `calbusbw.py` | Selects best algo from profile, returns intra/inter BW |
| Modify flow dependencies | `ring.py` / `tree.py` / `nvls.py` | `parent_flow_ids` / `child_flow_ids` on P2PFlow |
| Implement collnet | `collnet.py` + `decompose.py` | Only remaining stubs |

## ALGORITHM STATUS

| Algorithm | Collective Types | Status | Topology Source |
|-----------|-----------------|--------|-----------------|
| ring | AllReduce, AllGather, ReduceScatter, AllToAll | Implemented | MockNcclGroup (fallback: sequential) |
| tree | AllReduce | Implemented | MockNcclGroup binary tree |
| nvls | AllReduce | Implemented | Star topology via virtual NVSwitch node |
| nvls_tree | AllReduce | Implemented | Intra-node NVLS + inter-node tree |
| collnet_direct | AllReduce | Stub | — |
| collnet_chain | AllReduce | Stub | — |

## CONVENTIONS

- **Protocol-based polymorphism** — `CCLDecomposer` is `typing.Protocol`, not ABC. `NCCLDecomposer` delegates to `decompose_collective()`
- **Algorithm registry** — `_REGISTRY` in `decompose.py` maps `(algorithm, collective_type)` → function
- **Flow IDs globally unique** — `flow_id_start` parameter ensures no collisions across decompositions
- **Multi-channel** — `num_channels` splits data into parallel independent flow sets
- **MockNcclGroup dependency** — tree and nvls require the C++ pybind11 extension; ring has sequential fallback
- **Back-fill child_flow_ids** — all algorithms build parent_flow_ids first, then derive child_flow_ids in a final pass
- **Virtual switch nodes** — NVLS uses IDs starting at `_SWITCH_BASE = 1_000_000` to avoid collision with GPU ranks

## ANTI-PATTERNS
- **CollNet is the only remaining stub** — collnet_direct and collnet_chain raise `NotImplementedError`
- **RCCLDecomposer is a stub** — raises `NotImplementedError`. Only `NCCLDecomposer` works
- **tree/nvls require C++ extension** — will raise `RuntimeError` if `simulon._mocknccl` not built
- **NVLS only for intra-node** — applying nvls to inter-node groups is wrong; calbusbw enforces this for auto-selection
