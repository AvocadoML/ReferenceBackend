/*
 * gemms.cpp
 *
 *  Created on: Nov 12, 2021
 *      Author: Maciej Kozarzewski
 */

#include <avocado/reference_backend.h>
#include <avocado/backend/backend_descriptors.hpp>
#include "utils.hpp"
#include "descriptors.hpp"

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
		avStatus_t refGemm(avContextDescriptor_t context, avGemmOperation_t aOp, avGemmOperation_t bOp, const void *alpha,
				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem,
				const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
			const avSize_t M = getTensor(cDesc).firstDim();
			const avSize_t N = getTensor(cDesc).lastDim();
			const avSize_t K = (aOp == AVOCADO_GEMM_OPERATION_N) ? getTensor(aDesc).lastDim() : getTensor(aDesc).firstDim();

			switch (getTensor(cDesc).dtype())
			{
				case AVOCADO_DTYPE_INT32:
				{
					if (getTensor(aDesc).dtype() != AVOCADO_DTYPE_INT8)
						return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
					kernel_gemm(aOp, bOp, getPointer<int32_t>(cMem), getPointer<int8_t>(aMem), getPointer<int8_t>(bMem),
							getAlphaValue<int32_t>(alpha), getBetaValue<int32_t>(beta), M, N, K);
					break;
				}
				case AVOCADO_DTYPE_BFLOAT16:
					kernel_gemm(aOp, bOp, getPointer<bfloat16>(cMem), getPointer<bfloat16>(aMem), getPointer<bfloat16>(bMem), getAlphaValue(alpha),
							getBetaValue(beta), M, N, K);
					break;
				case AVOCADO_DTYPE_FLOAT16:
					kernel_gemm(aOp, bOp, getPointer<float16>(cMem), getPointer<float16>(aMem), getPointer<float16>(bMem), getAlphaValue(alpha),
							getBetaValue(beta), M, N, K);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					kernel_gemm(aOp, bOp, getPointer<float>(cMem), getPointer<float>(aMem), getPointer<float>(bMem), getAlphaValue(alpha),
							getBetaValue(beta), M, N, K);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_gemm(aOp, bOp, getPointer<double>(cMem), getPointer<double>(aMem), getPointer<double>(bMem), getAlphaValue<double>(alpha),
							getBetaValue<double>(beta), M, N, K);
					break;
				case AVOCADO_DTYPE_COMPLEX32:
					kernel_gemm(aOp, bOp, getPointer<std::complex<float>>(cMem), getPointer<std::complex<float>>(aMem),
							getPointer<std::complex<float>>(bMem), getAlphaValue<std::complex<float>>(alpha), getBetaValue<std::complex<float>>(beta),
							M, N, K);
					break;
				case AVOCADO_DTYPE_COMPLEX64:
					kernel_gemm(aOp, bOp, getPointer<std::complex<double>>(cMem), getPointer<std::complex<double>>(aMem),
							getPointer<std::complex<double>>(bMem), getAlphaValue<std::complex<double>>(alpha),
							getBetaValue<std::complex<double>>(beta), M, N, K);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refGemmBatched(avContextDescriptor_t context, avGemmOperation_t aOp, avGemmOperation_t bOp, const void *alpha,
				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem,
				const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
			const avSize_t batch = getTensor(cDesc).dimension(0);
			const avSize_t M = getTensor(cDesc).dimension(1);
			const avSize_t N = getTensor(cDesc).dimension(2);
			const avSize_t K = (aOp == AVOCADO_GEMM_OPERATION_N) ? getTensor(aDesc).dimension(2) : getTensor(aDesc).dimension(1);

			for (avSize_t b = 0; b < batch; b++)
			{
				avSize_t c_offset = b * M * N;
				avSize_t a_offset = b * M * K;
				avSize_t b_offset = b * N * K;
				switch (getTensor(cDesc).dtype())
				{
					case AVOCADO_DTYPE_INT32:
					{
						if (getTensor(aDesc).dtype() != AVOCADO_DTYPE_INT8)
							return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
						kernel_gemm(aOp, bOp, getPointer<int32_t>(cMem) + c_offset, getPointer<int8_t>(aMem) + a_offset,
								getPointer<int8_t>(bMem) + b_offset, getAlphaValue<int32_t>(alpha), getBetaValue<int32_t>(beta), M, N, K);
						break;
					}
					case AVOCADO_DTYPE_BFLOAT16:
						kernel_gemm(aOp, bOp, getPointer<bfloat16>(cMem) + c_offset, getPointer<bfloat16>(aMem) + a_offset,
								getPointer<bfloat16>(bMem) + b_offset, getAlphaValue(alpha), getBetaValue(beta), M, N, K);
						break;
					case AVOCADO_DTYPE_FLOAT16:
						kernel_gemm(aOp, bOp, getPointer<float16>(cMem) + c_offset, getPointer<float16>(aMem) + a_offset,
								getPointer<float16>(bMem) + b_offset, getAlphaValue(alpha), getBetaValue(beta), M, N, K);
						break;
					case AVOCADO_DTYPE_FLOAT32:
						kernel_gemm(aOp, bOp, getPointer<float>(cMem) + c_offset, getPointer<float>(aMem) + a_offset,
								getPointer<float>(bMem) + b_offset, getAlphaValue(alpha), getBetaValue(beta), M, N, K);
						break;
					case AVOCADO_DTYPE_FLOAT64:
						kernel_gemm(aOp, bOp, getPointer<double>(cMem) + c_offset, getPointer<double>(aMem) + a_offset,
								getPointer<double>(bMem) + b_offset, getAlphaValue<double>(alpha), getBetaValue<double>(beta), M, N, K);
						break;
					case AVOCADO_DTYPE_COMPLEX32:
						kernel_gemm(aOp, bOp, getPointer<std::complex<float>>(cMem) + c_offset, getPointer<std::complex<float>>(aMem) + a_offset,
								getPointer<std::complex<float>>(bMem) + b_offset, getAlphaValue<std::complex<float>>(alpha),
								getBetaValue<std::complex<float>>(beta), M, N, K);
						break;
					case AVOCADO_DTYPE_COMPLEX64:
						kernel_gemm(aOp, bOp, getPointer<std::complex<double>>(cMem) + c_offset, getPointer<std::complex<double>>(aMem) + a_offset,
								getPointer<std::complex<double>>(bMem) + b_offset, getAlphaValue<std::complex<double>>(alpha),
								getBetaValue<std::complex<double>>(beta), M, N, K);
						break;
					default:
						return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
				}
			}
			return AVOCADO_STATUS_SUCCESS;
		}

	} /* namespace backend */
} /* namespace avocado */

