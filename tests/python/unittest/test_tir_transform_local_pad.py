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
from tvm.tir.transform.transform import (
    LowerInitBlock,
    PlanAndUpdateBufferAllocationLocation,
    ConvertBlocksToOpaque,
    CompactBufferAllocation,
    FlattenBuffer,
    Simplify,
    LocalPad,
    VectorizeLoop,
    InjectVirtualThread,
    StorageRewrite,
)


@tvm.script.ir_module
class MatMulNNOriginalModule:
    """
    The original module generated by the MetaScheduler.
    """

    # fmt: off
    @T.prim_func
    def main(A: T.Buffer[(960, 770), "float32"], B: T.Buffer[(770, 2304), "float32"], C: T.Buffer[(960, 2304), "float32"]) -> None:
        C_local = T.alloc_buffer([960, 2304], dtype="float32", scope="local")
        A_shared = T.alloc_buffer([960, 770], dtype="float32", scope="shared")
        B_shared = T.alloc_buffer([770, 2304], dtype="float32", scope="shared")
        for i_0_j_0_fused in T.thread_binding(144, thread="blockIdx.x"):
            for i_1_j_1_fused in T.thread_binding(2, thread="vthread.x"):
                for i_2_j_2_fused in T.thread_binding(256, thread="threadIdx.x"):
                    for i_3_init, j_3_init, i_4_init, j_4_init in T.grid(8, 1, 2, 2):
                        with T.block("update_init"):
                            vi = T.axis.spatial(960, (((i_0_j_0_fused // 18 + 0) * 8 + i_2_j_2_fused // 32) * 8 + i_3_init) * 2 + i_4_init)
                            vj = T.axis.spatial(2304, ((i_0_j_0_fused % 18 * 2 + i_1_j_1_fused % 2) * 32 + i_2_j_2_fused % 32 + j_3_init) * 2 + j_4_init)
                            T.where((((i_0_j_0_fused // 18 + 0) * 8 + i_2_j_2_fused // 32) * 8 + i_3_init) * 2 + i_4_init < 960)
                            T.reads()
                            T.writes(C_local[vi, vj])
                            C_local[vi, vj] = T.float32(0)
                    for k_0 in T.serial(193):
                        for ax0_ax1_fused_0 in T.serial(1):
                            for ax0_ax1_fused_1 in T.thread_binding(256, thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(3):
                                    with T.block("A_shared"):
                                        v0 = T.axis.spatial(960, i_0_j_0_fused // 18 * 128 + (ax0_ax1_fused_0 * 768 + ax0_ax1_fused_1 * 3 + ax0_ax1_fused_2) // 4)
                                        v1 = T.axis.spatial(770, k_0 * 4 + (ax0_ax1_fused_0 * 768 + ax0_ax1_fused_1 * 3 + ax0_ax1_fused_2) % 4)
                                        T.where(i_0_j_0_fused // 18 * 128 + ((ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1) * 3 + ax0_ax1_fused_2) // 4 < 960 and k_0 * 4 + ((ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1) * 3 + ax0_ax1_fused_2) % 4 < 770 and (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1) * 3 + ax0_ax1_fused_2 < 512)
                                        T.reads(A[v0, v1])
                                        T.writes(A_shared[v0, v1])
                                        A_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused_0 in T.serial(1):
                            for ax0_ax1_fused_1 in T.thread_binding(256, thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(4):
                                    with T.block("B_shared"):
                                        v0 = T.axis.spatial(770, k_0 * 4 + (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 4 + ax0_ax1_fused_2) // 128)
                                        v1 = T.axis.spatial(2304, i_0_j_0_fused % 18 * 128 + (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 4 + ax0_ax1_fused_2) % 128)
                                        T.where(k_0 * 4 + ((ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1) * 4 + ax0_ax1_fused_2) // 128 < 770 and (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1) * 4 + ax0_ax1_fused_2 < 512)
                                        T.reads(B[v0, v1])
                                        T.writes(B_shared[v0, v1])
                                        B_shared[v0, v1] = B[v0, v1]
                        for k_1, i_3, j_3, k_2, i_4, j_4 in T.grid(1, 8, 1, 4, 2, 2):
                            with T.block("update_update"):
                                vi = T.axis.spatial(960, (((i_0_j_0_fused // 18 + 0) * 8 + i_2_j_2_fused // 32) * 8 + i_3) * 2 + i_4)
                                vj = T.axis.spatial(2304, ((i_0_j_0_fused % 18 * 2 + i_1_j_1_fused % 2) * 32 + i_2_j_2_fused % 32 + j_3) * 2 + j_4)
                                vk = T.axis.reduce(770, (k_0 + k_1) * 4 + k_2)
                                T.where((((i_0_j_0_fused // 18 + 0) * 8 + i_2_j_2_fused // 32) * 8 + i_3) * 2 + i_4 < 960 and (k_0 + k_1) * 4 + k_2 < 770)
                                T.reads(C_local[vi, vj], A_shared[vi, vk], B_shared[vk, vj])
                                T.writes(C_local[vi, vj])
                                C_local[vi, vj] = C_local[vi, vj] + A_shared[vi, vk] * B_shared[vk, vj]
                    for ax0, ax1 in T.grid(16, 2):
                        with T.block("C_local"):
                            v0 = T.axis.spatial(960, i_0_j_0_fused // 18 * 128 + i_2_j_2_fused // 32 * 16 + ax0)
                            v1 = T.axis.spatial(2304, i_0_j_0_fused % 18 * 128 + i_1_j_1_fused * 64 + i_2_j_2_fused % 32 * 2 + ax1)
                            T.where(i_0_j_0_fused // 18 * 128 + i_2_j_2_fused // 32 * 16 + ax0 < 960)
                            T.reads(C_local[v0, v1])
                            T.writes(C[v0, v1])
                            C[v0, v1] = C_local[v0, v1]

    # fmt: on


@tvm.script.ir_module
class MatMulNNExpectedModule:
    """
    The expected module generated from the local padding pass.
    """

    # fmt: off
    @T.prim_func
    def main(A: T.Buffer[(960, 770), "float32"], B: T.Buffer[(770, 2304), "float32"], C: T.Buffer[(960, 2304), "float32"]) -> None:
        C_local = T.alloc_buffer([960, 2304], dtype="float32", scope="local")
        A_shared = T.alloc_buffer([960, 770], dtype="float32", scope="shared")
        B_shared = T.alloc_buffer([770, 2304], dtype="float32", scope="shared")
        for i_0_j_0_fused in T.thread_binding(144, thread="blockIdx.x"):
            for i_1_j_1_fused in T.thread_binding(2, thread="vthread.x"):
                for i_2_j_2_fused in T.thread_binding(256, thread="threadIdx.x"):
                    for i_3_init, j_3_init, i_4_init, j_4_init in T.grid(8, 1, 2, 2):
                        with T.block("update_init"):
                            vi = T.axis.spatial(960, (((i_0_j_0_fused // 18 + 0) * 8 + i_2_j_2_fused // 32) * 8 + i_3_init) * 2 + i_4_init)
                            vj = T.axis.spatial(2304, ((i_0_j_0_fused % 18 * 2 + i_1_j_1_fused % 2) * 32 + i_2_j_2_fused % 32 + j_3_init) * 2 + j_4_init)
                            T.reads()
                            T.writes(C_local[vi, vj])
                            C_local[vi, vj] = T.float32(0)
                    for k_0 in T.serial(193):
                        for ax0_ax1_fused_0 in T.serial(1):
                            for ax0_ax1_fused_1 in T.thread_binding(256, thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(3):
                                    with T.block("A_shared"):
                                        v0 = T.axis.spatial(960, i_0_j_0_fused // 18 * 128 + (ax0_ax1_fused_0 * 768 + ax0_ax1_fused_1 * 3 + ax0_ax1_fused_2) // 4)
                                        v1 = T.axis.spatial(770, k_0 * 4 + (ax0_ax1_fused_0 * 768 + ax0_ax1_fused_1 * 3 + ax0_ax1_fused_2) % 4)
                                        T.where(i_0_j_0_fused // 18 * 128 + ((ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1) * 3 + ax0_ax1_fused_2) // 4 < 960 and k_0 * 4 + ((ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1) * 3 + ax0_ax1_fused_2) % 4 < 770 and (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1) * 3 + ax0_ax1_fused_2 < 512)
                                        T.reads(A[v0, v1])
                                        T.writes(A_shared[v0, v1])
                                        A_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused_0 in T.serial(1):
                            for ax0_ax1_fused_1 in T.thread_binding(256, thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(4):
                                    with T.block("B_shared"):
                                        v0 = T.axis.spatial(770, k_0 * 4 + (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 4 + ax0_ax1_fused_2) // 128)
                                        v1 = T.axis.spatial(2304, i_0_j_0_fused % 18 * 128 + (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 4 + ax0_ax1_fused_2) % 128)
                                        T.where(k_0 * 4 + ((ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1) * 4 + ax0_ax1_fused_2) // 128 < 770 and (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1) * 4 + ax0_ax1_fused_2 < 512)
                                        T.reads(B[v0, v1])
                                        T.writes(B_shared[v0, v1])
                                        B_shared[v0, v1] = B[v0, v1]
                        for k_1, i_3, j_3, k_2, i_4, j_4 in T.grid(1, 8, 1, 4, 2, 2):
                            with T.block("update_update"):
                                vi = T.axis.spatial(960, (((i_0_j_0_fused // 18 + 0) * 8 + i_2_j_2_fused // 32) * 8 + i_3) * 2 + i_4)
                                vj = T.axis.spatial(2304, ((i_0_j_0_fused % 18 * 2 + i_1_j_1_fused % 2) * 32 + i_2_j_2_fused % 32 + j_3) * 2 + j_4)
                                vk = T.axis.reduce(770, (k_0 + k_1) * 4 + k_2)
                                T.where((k_0 + k_1) * 4 + k_2 < 770)
                                T.reads(C_local[vi, vj], A_shared[vi, vk], B_shared[vk, vj])
                                T.writes(C_local[vi, vj])
                                C_local[vi, vj] = C_local[vi, vj] + A_shared[vi, vk] * B_shared[vk, vj]
                    for ax0, ax1 in T.grid(16, 2):
                        with T.block("C_local"):
                            v0 = T.axis.spatial(960, i_0_j_0_fused // 18 * 128 + i_2_j_2_fused // 32 * 16 + ax0)
                            v1 = T.axis.spatial(2304, i_0_j_0_fused % 18 * 128 + i_1_j_1_fused * 64 + i_2_j_2_fused % 32 * 2 + ax1)
                            T.where(i_0_j_0_fused // 18 * 128 + i_2_j_2_fused // 32 * 16 + ax0 < 960)
                            T.reads(C_local[v0, v1])
                            T.writes(C[v0, v1])
                            C[v0, v1] = C_local[v0, v1]

    # fmt: on


def preprocess(mod, enable_local_pad):
    """
    Pre-process the IRModule by lowering the BlockScope's.
    """
    mod = LowerInitBlock()(mod)
    mod = PlanAndUpdateBufferAllocationLocation()(mod)
    mod = ConvertBlocksToOpaque()(mod)
    mod = CompactBufferAllocation(enable_local_pad)(mod)
    mod = Simplify()(mod)
    return mod


def postprocess(mod):
    """
    Post-process the IRModule. This is to ensure that the shared memory variables are ordered.
    """
    mod = FlattenBuffer()(mod)
    mod = VectorizeLoop(True)(mod)
    mod = InjectVirtualThread()(mod)
    mod = StorageRewrite()(mod)
    return mod


def test_dense_local_padding():
    """
    Test that local padding is delivering the correct compute outcome.
    """
    # The workload is deliberately selected so that it does not fit into the sample schedule.
    mod = MatMulNNOriginalModule
    mod = preprocess(mod, True)
    mod = LocalPad(True)(mod)
    expected_mod = MatMulNNExpectedModule
    expected_mod = preprocess(expected_mod, False)
    tvm.ir.assert_structural_equal(postprocess(mod), postprocess(expected_mod))


if __name__ == "__main__":
    test_dense_local_padding()
