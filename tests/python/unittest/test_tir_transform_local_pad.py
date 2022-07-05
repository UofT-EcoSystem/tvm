# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=missing-module-docstring
from tvm.script import tir as T
import tvm.testing
from tvm.tir import Schedule
from tvm.tir.transform.transform import LowerInitBlock, PlanAndUpdateBufferAllocationLocation, \
                                        ConvertBlocksToOpaque, CompactBufferAllocation, \
                                        FlattenBuffer, Simplify, LocalPad, VectorizeLoop


def sample_dense_sched(sch):  # pylint: disable=too-many-statements
    """
    Given below is a sample schedule generated by the MetaScheduler.
    """
    # pylint: disable=unused-variable, invalid-name. too-many-locals
    b0 = sch.get_block(name="update", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    l2, l3, l4 = sch.get_loops(block=b0)
    l10, l11, l12, l13, l14 = sch.split(loop=l2, factors=[None, 1, 8, 8, 2])
    l20, l21, l22, l23, l24 = sch.split(loop=l3, factors=[None, 2, 32, 1, 2])
    l28, l29, l30 = sch.split(loop=l4, factors=[None, 1, 4])
    sch.reorder(l10, l20, l11, l21, l12, l22, l28, l29, l13, l23, l30, l14, l24)
    l31 = sch.fuse(l10, l20)
    sch.bind(loop=l31, thread_axis="blockIdx.x")
    l32 = sch.fuse(l11, l21)
    sch.bind(loop=l32, thread_axis="vthread.x")
    l33 = sch.fuse(l12, l22)
    sch.bind(loop=l33, thread_axis="threadIdx.x")
    b34 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b34, loop=l33, preserve_unit_loops=True)
    b35 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="shared")
    sch.compute_at(block=b35, loop=l28, preserve_unit_loops=True)
    l36, l37, l38, l39, l40, l41 = sch.get_loops(block=b35)
    l42 = sch.fuse(l40, l41)
    sch.annotate(block_or_loop=b35, ann_key="meta_schedule.cooperative_fetch", ann_val=3)
    b44 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared")
    sch.compute_at(block=b44, loop=l28, preserve_unit_loops=True)
    l45, l46, l47, l48, l49, l50 = sch.get_loops(block=b44)
    l51 = sch.fuse(l49, l50)
    sch.annotate(block_or_loop=b44, ann_key="meta_schedule.cooperative_fetch", ann_val=4)
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=512)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b35, ann_key="meta_schedule.cooperative_fetch")
    l54, l55, l56, l57, l58 = sch.get_loops(block=b35)
    l59, l60, l61 = sch.split(loop=l58, factors=[None, 256, 3])
    sch.vectorize(loop=l61)
    sch.bind(loop=l60, thread_axis="threadIdx.x")
    sch.unannotate(block_or_loop=b44, ann_key="meta_schedule.cooperative_fetch")
    l61, l62, l63, l64, l65 = sch.get_loops(block=b44)
    l66, l67, l68 = sch.split(loop=l65, factors=[None, 256, 4])
    sch.vectorize(loop=l68)
    sch.bind(loop=l67, thread_axis="threadIdx.x")
    b69 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b69, ann_key="meta_schedule.unroll_explicit")
    b70, b71, b72, b73 = sch.get_child_blocks(b69)
    l74, l75, l76, l77, l78, l79, l80 = sch.get_loops(block=b70)
    sch.annotate(block_or_loop=l74, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l74, ann_key="pragma_unroll_explicit", ann_val=1)
    l80, l81, l82, l83, l84, l85, l86 = sch.get_loops(block=b71)
    sch.annotate(block_or_loop=l80, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l80, ann_key="pragma_unroll_explicit", ann_val=1)
    l87, l88, l89, l90, l91, l92, l93, l94, l95, l96 = sch.get_loops(block=b72)
    sch.annotate(block_or_loop=l87, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l87, ann_key="pragma_unroll_explicit", ann_val=1)
    l97, l98, l99, l100, l101 = sch.get_loops(block=b73)
    sch.annotate(block_or_loop=l97, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l97, ann_key="pragma_unroll_explicit", ann_val=1)
    b102 = sch.get_block(name="update", func_name="main")
    l103, l104, l105, l106, l107, l108, l109, l110, l111, l112 = sch.get_loops(block=b102)
    b113 = sch.decompose_reduction(block=b102, loop=l106)
    # pylint: enable=unused-variable, invalid-name. too-many-locals, too-many-statements


def MatMulNN(M: int, K: int, N: int):
    # pylint: disable=invalid-name, no-member, missing-function-docstring
    @T.prim_func
    def wkl_func(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, [M, K])
        B = T.match_buffer(b, [K, N])
        C = T.match_buffer(c, [M, N])
        for i, j, k in T.grid(M, N, K):
            with T.block("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
    return wkl_func
    # pylint: enable=invalid-name, no-member, missing-function-docstring


@tvm.script.ir_module
class MatMulNNExpectedModule:
    """
    The expected module generated from the local padding pass.
    """
    # pylint: disable=invalid-name, no-member, missing-function-docstring, no-self-argument, too-many-locals, too-many-nested-blocks, too-many-branches, too-few-public-methods
    @T.prim_func
    def main(A: T.Buffer[(960, 770), "float32"], B: T.Buffer[(770, 2304), "float32"],
             C: T.Buffer[(960, 2304), "float32"]) -> None:
        C_local = T.alloc_buffer([960, 2304], dtype="float32", scope="local")
        A_shared = T.alloc_buffer([960, 770], dtype="float32", scope="shared")
        B_shared = T.alloc_buffer([770, 2304], dtype="float32", scope="shared")
        for i_0_j_0_fused in \
                T.thread_binding(144, thread="blockIdx.x",
                                 annotations={"pragma_auto_unroll_max_step":512,
                                              "pragma_unroll_explicit":1}):
            for i_1_j_1_fused in T.thread_binding(2, thread="vthread.x"):
                for i_2_j_2_fused in T.thread_binding(256, thread="threadIdx.x"):
                    for i_3_init, j_3_init, i_4_init, j_4_init in T.grid(8, 1, 2, 2):
                        with T.block("update_init"):
                            vi = T.axis.spatial(960,
                                (((i_0_j_0_fused // 18 + 0) * 8 + i_2_j_2_fused // 32) * 8 +
                                 i_3_init) * 2 + i_4_init
                            )
                            vj = T.axis.spatial(2304,
                                ((i_0_j_0_fused % 18 * 2 + i_1_j_1_fused % 2) * 32 +
                                 i_2_j_2_fused % 32 + j_3_init) * 2 + j_4_init
                            )
                            T.reads()
                            T.writes(C_local[vi, vj])
                            T.block_attr({"meta_schedule.tiling_structure":"SSSRRSRS"})
                            C_local[vi, vj] = T.float32(0)
                    for k_0 in T.serial(193):
                        for ax0_ax1_fused_0 in T.serial(1):
                            for ax0_ax1_fused_1 in T.thread_binding(256, thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(3):
                                    with T.block("A_shared"):
                                        v0 = T.axis.spatial(960,
                                            i_0_j_0_fused // 18 * 128 +
                                            (ax0_ax1_fused_0 * 768 + ax0_ax1_fused_1 * 3 +
                                             ax0_ax1_fused_2) // 4
                                        )
                                        v1 = T.axis.spatial(770,
                                            k_0 * 4 +
                                            (ax0_ax1_fused_0 * 768 + ax0_ax1_fused_1 * 3 +
                                             ax0_ax1_fused_2) % 4
                                        )
                                        T.where((ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1) * 3 +
                                                ax0_ax1_fused_2 < 512)
                                        T.reads(A[v0, v1])
                                        T.writes(A_shared[v0, v1])
                                        A_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused_0 in T.serial(1):
                            for ax0_ax1_fused_1 in T.thread_binding(256, thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(4):
                                    with T.block("B_shared"):
                                        v0 = T.axis.spatial(770,
                                            k_0 * 4 +
                                            (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 4 +
                                             ax0_ax1_fused_2) // 128
                                        )
                                        v1 = T.axis.spatial(2304,
                                            i_0_j_0_fused % 18 * 128 +
                                            (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 4 +
                                             ax0_ax1_fused_2) % 128
                                        )
                                        T.where((ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1) * 4 +
                                                ax0_ax1_fused_2 < 512)
                                        T.reads(B[v0, v1])
                                        T.writes(B_shared[v0, v1])
                                        B_shared[v0, v1] = B[v0, v1]
                        for k_1, i_3, j_3, k_2, i_4, j_4 in T.grid(1, 8, 1, 4, 2, 2):
                            with T.block("update_update"):
                                vi = T.axis.spatial(960,
                                    (((i_0_j_0_fused // 18 + 0) * 8 + i_2_j_2_fused // 32) * 8 +
                                     i_3) * 2 + i_4
                                )
                                vj = T.axis.spatial(2304,
                                    ((i_0_j_0_fused % 18 * 2 + i_1_j_1_fused % 2) * 32 +
                                     i_2_j_2_fused % 32 + j_3) * 2 + j_4
                                )
                                vk = T.axis.reduce(770, (k_0 + k_1) * 4 + k_2)
                                T.reads(C_local[vi, vj], A_shared[vi, vk], B_shared[vk, vj])
                                T.writes(C_local[vi, vj])
                                T.block_attr({"meta_schedule.tiling_structure":"SSSRRSRS"})
                                C_local[vi, vj] = \
                                        C_local[vi, vj] + A_shared[vi, vk] * B_shared[vk, vj]
                    for ax0, ax1 in T.grid(16, 2):
                        with T.block("C_local"):
                            v0 = T.axis.spatial(960,
                                i_0_j_0_fused // 18 * 128 + i_2_j_2_fused // 32 * 16 + ax0
                            )
                            v1 = T.axis.spatial(2304,
                                i_0_j_0_fused % 18 * 128 + i_1_j_1_fused * 64 +
                                i_2_j_2_fused % 32 * 2 + ax1
                            )
                            T.where(i_0_j_0_fused // 18 * 128 + i_2_j_2_fused // 32 * 16 +
                                    ax0 < 960)
                            T.reads(C_local[v0, v1])
                            T.writes(C[v0, v1])
                            C[v0, v1] = C_local[v0, v1]
    # pylint: enable=invalid-name, no-member, missing-function-docstring, no-self-argument, too-many-locals, too-many-nested-blocks, too-many-branches, too-few-public-methods


def preprocess(mod):
    mod = LowerInitBlock()(mod)
    mod = PlanAndUpdateBufferAllocationLocation()(mod)
    mod = ConvertBlocksToOpaque()(mod)
    mod = CompactBufferAllocation()(mod)

    print(mod)

    mod = FlattenBuffer()(mod)
    mod = Simplify()(mod)
    return mod


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_dense_local_padding():
    """
    Test that local padding is delivering the correct compute outcome.
    """
    # The workload is deliberately selected so that it does not fit into the sample schedule.
    tir_sched = Schedule(MatMulNN(960, 770, 2304))
    sample_dense_sched(tir_sched)

    print(tir_sched.mod.script())

    mod = tir_sched.mod
    
    mod = preprocess(mod)
    mod = LocalPad(True)(mod)
    mod = VectorizeLoop(False, True)(mod)

    # print(mod)
    preprocess(MatMulNNExpectedModule)

    # print(tir_sched.mod["main"])
    # print(tvm.lower(tir_sched.mod["main"], []))


if __name__ == "__main__":
    test_dense_local_padding()
