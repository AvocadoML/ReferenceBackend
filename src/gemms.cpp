/*
 * gemms.cpp
 *
 *  Created on: Nov 12, 2021
 *      Author: Maciej Kozarzewski
 */

#include <avocado/reference_backend.h>
#include <avocado/backend/backend_descriptors.hpp>
#include "utils.hpp"

namespace
{
	using namespace avocado::backend;
	template<typename C_type, typename AB_type, typename Compute_type>
	void kernel_gemm(avGemmOperation_t opA, avGemmOperation_t opB, C_type *C, const AB_type *A, const AB_type *B, Compute_type alpha,
			Compute_type beta, avSize_t M, avSize_t N, avSize_t K)
	{
		if (beta == zero<Compute_type>())
			clear(C, M * N);
		for (avSize_t m = 0; m < M; m++)
			for (avSize_t n = 0; n < N; n++)
			{
				Compute_type tmp = zero<Compute_type>();
				if (opA == AVOCADO_GEMM_OPERATION_N)
				{
					if (opB == AVOCADO_GEMM_OPERATION_N)
					{
						for (avSize_t k = 0; k < K; k++) // C (M x N) = A (M x K) * B (K x N)
							tmp += static_cast<Compute_type>(A[m * K + k]) * static_cast<Compute_type>(B[k * N + n]);
					}
					else // B is transposed
					{
						for (avSize_t k = 0; k < K; k++) // C (M x N) = A (M x K) * B (N x K)
							tmp += static_cast<Compute_type>(A[m * K + k]) * static_cast<Compute_type>(B[n * K + k]);
					}
				}
				else // A is transposed
				{
					if (opB == AVOCADO_GEMM_OPERATION_N)
					{
						for (avSize_t k = 0; k < K; k++) // C (M x N) = A (K x N) * B (K x N)
							tmp += static_cast<Compute_type>(A[k * N + n]) * static_cast<Compute_type>(B[k * N + n]);
					}
					else // B is transposed
					{
						for (avSize_t k = 0; k < K; k++) // C (M x N) = A (K x M) * B (N x K)
							tmp += static_cast<Compute_type>(A[k * N + n]) * static_cast<Compute_type>(B[n * K + k]);
					}
				}
				C[m * N + n] = alpha * tmp + beta * static_cast<Compute_type>(C[m * N + n]);
			}
	}
}
namespace avocado
{
	namespace backend
	{
//		avStatus_t refGemm(avContext_t context, avGemmOperation_t opA, avGemmOperation_t opB, avTensor_t C, const avTensor_t A, const avTensor_t B,
//				const avScalar_t alpha, const avScalar_t beta)
//		{
//			assert(context != nullptr);
//			assert(C != nullptr);
//			assert(A != nullptr);
//			assert(B != nullptr);
//			assert(same_type(A, B));
//
//			const avSize_t M = dimension(C, 0);
//			const avSize_t N = dimension(C, 1);
//			const avSize_t K = (opA == AVOCADO_GEMM_OPERATION_N) ? dimension(A, 1) : dimension(A, 0);
//
//			switch (C->dtype)
//			{
//				case AVOCADO_DTYPE_INT32:
//				{
//					if (A->dtype != AVOCADO_DTYPE_INT8)
//						return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
//					kernel_gemm(opA, opB, data<int32_t>(C), data<int8_t>(A), data<int8_t>(B), getAlphaValue<int32_t>(alpha),
//							getBetaValue<int32_t>(beta), M, N, K);
//					break;
//				}
//				case AVOCADO_DTYPE_BFLOAT16:
//					kernel_gemm(opA, opB, data<bfloat16>(C), data<bfloat16>(A), data<bfloat16>(B), getAlphaValue<float>(alpha),
//							getBetaValue<float>(beta), M, N, K);
//					break;
//				case AVOCADO_DTYPE_FLOAT16:
//					kernel_gemm(opA, opB, data<float16>(C), data<float16>(A), data<float16>(B), getAlphaValue<float>(alpha),
//							getBetaValue<float>(beta), M, N, K);
//					break;
//				case AVOCADO_DTYPE_FLOAT32:
//					kernel_gemm(opA, opB, data<float>(C), data<float>(A), data<float>(B), getAlphaValue<float>(alpha), getBetaValue<float>(beta), M,
//							N, K);
//					break;
//				case AVOCADO_DTYPE_FLOAT64:
//					kernel_gemm(opA, opB, data<double>(C), data<double>(A), data<double>(B), getAlphaValue<double>(alpha), getBetaValue<double>(beta),
//							M, N, K);
//					break;
//				case AVOCADO_DTYPE_COMPLEX32:
//					kernel_gemm(opA, opB, data<std::complex<float>>(C), data<std::complex<float>>(A), data<std::complex<float>>(B),
//							getAlphaValue<std::complex<float>>(alpha), getBetaValue<std::complex<float>>(beta), M, N, K);
//					break;
//				case AVOCADO_DTYPE_COMPLEX64:
//					kernel_gemm(opA, opB, data<std::complex<double>>(C), data<std::complex<double>>(A), data<std::complex<double>>(B),
//							getAlphaValue<std::complex<double>>(alpha), getBetaValue<std::complex<double>>(beta), M, N, K);
//					break;
//				default:
//					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
//			}
//			return AVOCADO_STATUS_SUCCESS;
//		}
//		avStatus_t refGemmBatched(avContext_t context, avGemmOperation_t opA, avGemmOperation_t opB, avTensor_t C, const avTensor_t A,
//				const avTensor_t B, const avScalar_t alpha, const avScalar_t beta)
//		{
//			assert(context != nullptr);
//			assert(C != nullptr);
//			assert(A != nullptr);
//			assert(B != nullptr);
//			assert(same_type(A, B));
//
//			const avSize_t batch = dimension(C, 0);
//			const avSize_t M = dimension(C, 1);
//			const avSize_t N = dimension(C, 2);
//			const avSize_t K = (opA == AVOCADO_GEMM_OPERATION_N) ? dimension(A, 2) : dimension(A, 1);
//
//			for (avSize_t b = 0; b < batch; b++)
//			{
//				avSize_t c_offset = b * M * N;
//				avSize_t a_offset = b * M * K;
//				avSize_t b_offset = b * N * K;
//				switch (C->dtype)
//				{
//					case AVOCADO_DTYPE_INT32:
//					{
//						if (A->dtype != AVOCADO_DTYPE_INT8)
//							return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
//						kernel_gemm(opA, opB, data<int32_t>(C) + c_offset, data<int8_t>(A) + a_offset, data<int8_t>(B) + b_offset,
//								getAlphaValue<int32_t>(alpha), getBetaValue<int32_t>(beta), M, N, K);
//						break;
//					}
//					case AVOCADO_DTYPE_BFLOAT16:
//						kernel_gemm(opA, opB, data<bfloat16>(C) + c_offset, data<bfloat16>(A) + a_offset, data<bfloat16>(B) + b_offset,
//								getAlphaValue<float>(alpha), getBetaValue<float>(beta), M, N, K);
//						break;
//					case AVOCADO_DTYPE_FLOAT16:
//						kernel_gemm(opA, opB, data<float16>(C) + c_offset, data<float16>(A) + a_offset, data<float16>(B) + b_offset,
//								getAlphaValue<float>(alpha), getBetaValue<float>(beta), M, N, K);
//						break;
//					case AVOCADO_DTYPE_FLOAT32:
//						kernel_gemm(opA, opB, data<float>(C) + c_offset, data<float>(A) + a_offset, data<float>(B) + b_offset,
//								getAlphaValue<float>(alpha), getBetaValue<float>(beta), M, N, K);
//						break;
//					case AVOCADO_DTYPE_FLOAT64:
//						kernel_gemm(opA, opB, data<double>(C) + c_offset, data<double>(A) + a_offset, data<double>(B) + b_offset,
//								getAlphaValue<double>(alpha), getBetaValue<double>(beta), M, N, K);
//						break;
//					case AVOCADO_DTYPE_COMPLEX32:
//						kernel_gemm(opA, opB, data<std::complex<float>>(C) + c_offset, data<std::complex<float>>(A) + a_offset,
//								data<std::complex<float>>(B) + b_offset, getAlphaValue<std::complex<float>>(alpha),
//								getBetaValue<std::complex<float>>(beta), M, N, K);
//						break;
//					case AVOCADO_DTYPE_COMPLEX64:
//						kernel_gemm(opA, opB, data<std::complex<double>>(C) + c_offset, data<std::complex<double>>(A) + a_offset,
//								data<std::complex<double>>(B) + b_offset, getAlphaValue<std::complex<double>>(alpha),
//								getBetaValue<std::complex<double>>(beta), M, N, K);
//						break;
//					default:
//						return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
//				}
//			}
//			return AVOCADO_STATUS_SUCCESS;
//		}

	} /* namespace backend */
} /* namespace avocado */

