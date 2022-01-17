/*
 * gemms.cpp
 *
 *  Created on: Nov 12, 2021
 *      Author: Maciej Kozarzewski
 */

#include <backend_descriptors.hpp>
#include <ReferenceBackend/reference_backend.h>

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
		avStatus_t refGemm(avContextDescriptor_t context, avGemmOperation_t aOp, avGemmOperation_t bOp, const void *alpha,
				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem,
				const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
			const avSize_t M = reference::getTensor(cDesc).firstDim();
			const avSize_t N = reference::getTensor(cDesc).lastDim();
			const avSize_t K = (aOp == AVOCADO_GEMM_OPERATION_N) ? reference::getTensor(aDesc).lastDim() : reference::getTensor(aDesc).firstDim();

			switch (reference::getTensor(cDesc).dtype())
			{
				case AVOCADO_DTYPE_INT32:
				{
					if (reference::getTensor(aDesc).dtype() != AVOCADO_DTYPE_INT8)
						return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
					kernel_gemm(aOp, bOp, reference::getPointer<int32_t>(cMem), reference::getPointer<int8_t>(aMem), reference::getPointer<int8_t>(bMem),
							reference::getAlphaValue<int32_t>(alpha), reference::getBetaValue<int32_t>(beta), M, N, K);
					break;
				}
				case AVOCADO_DTYPE_BFLOAT16:
					kernel_gemm(aOp, bOp, reference::getPointer<bfloat16>(cMem), reference::getPointer<bfloat16>(aMem), reference::getPointer<bfloat16>(bMem), reference::getAlphaValue(alpha),
							reference::getBetaValue(beta), M, N, K);
					break;
				case AVOCADO_DTYPE_FLOAT16:
					kernel_gemm(aOp, bOp, reference::getPointer<float16>(cMem), reference::getPointer<float16>(aMem), reference::getPointer<float16>(bMem), reference::getAlphaValue(alpha),
							reference::getBetaValue(beta), M, N, K);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					kernel_gemm(aOp, bOp, reference::getPointer<float>(cMem), reference::getPointer<float>(aMem), reference::getPointer<float>(bMem), reference::getAlphaValue(alpha),
							reference::getBetaValue(beta), M, N, K);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_gemm(aOp, bOp, reference::getPointer<double>(cMem), reference::getPointer<double>(aMem), reference::getPointer<double>(bMem), reference::getAlphaValue<double>(alpha),
							reference::getBetaValue<double>(beta), M, N, K);
					break;
				case AVOCADO_DTYPE_COMPLEX32:
					kernel_gemm(aOp, bOp, reference::getPointer<std::complex<float>>(cMem), reference::getPointer<std::complex<float>>(aMem),
							reference::getPointer<std::complex<float>>(bMem), reference::getAlphaValue<std::complex<float>>(alpha), reference::getBetaValue<std::complex<float>>(beta),
							M, N, K);
					break;
				case AVOCADO_DTYPE_COMPLEX64:
					kernel_gemm(aOp, bOp, reference::getPointer<std::complex<double>>(cMem), reference::getPointer<std::complex<double>>(aMem),
							reference::getPointer<std::complex<double>>(bMem), reference::getAlphaValue<std::complex<double>>(alpha),
							reference::getBetaValue<std::complex<double>>(beta), M, N, K);
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
			const avSize_t batch = reference::getTensor(cDesc).dimension(0);
			const avSize_t M = reference::getTensor(cDesc).dimension(1);
			const avSize_t N = reference::getTensor(cDesc).dimension(2);
			const avSize_t K = (aOp == AVOCADO_GEMM_OPERATION_N) ? reference::getTensor(aDesc).dimension(2) : reference::getTensor(aDesc).dimension(1);

			for (avSize_t b = 0; b < batch; b++)
			{
				avSize_t c_offset = b * M * N;
				avSize_t a_offset = b * M * K;
				avSize_t b_offset = b * N * K;
				switch (reference::getTensor(cDesc).dtype())
				{
					case AVOCADO_DTYPE_INT32:
					{
						if (reference::getTensor(aDesc).dtype() != AVOCADO_DTYPE_INT8)
							return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
						kernel_gemm(aOp, bOp, reference::getPointer<int32_t>(cMem) + c_offset, reference::getPointer<int8_t>(aMem) + a_offset,
								reference::getPointer<int8_t>(bMem) + b_offset, reference::getAlphaValue<int32_t>(alpha), reference::getBetaValue<int32_t>(beta), M, N, K);
						break;
					}
					case AVOCADO_DTYPE_BFLOAT16:
						kernel_gemm(aOp, bOp, reference::getPointer<bfloat16>(cMem) + c_offset, reference::getPointer<bfloat16>(aMem) + a_offset,
								reference::getPointer<bfloat16>(bMem) + b_offset, reference::getAlphaValue(alpha), reference::getBetaValue(beta), M, N, K);
						break;
					case AVOCADO_DTYPE_FLOAT16:
						kernel_gemm(aOp, bOp, reference::getPointer<float16>(cMem) + c_offset, reference::getPointer<float16>(aMem) + a_offset,
								reference::getPointer<float16>(bMem) + b_offset, reference::getAlphaValue(alpha), reference::getBetaValue(beta), M, N, K);
						break;
					case AVOCADO_DTYPE_FLOAT32:
						kernel_gemm(aOp, bOp, reference::getPointer<float>(cMem) + c_offset, reference::getPointer<float>(aMem) + a_offset,
								reference::getPointer<float>(bMem) + b_offset, reference::getAlphaValue(alpha), reference::getBetaValue(beta), M, N, K);
						break;
					case AVOCADO_DTYPE_FLOAT64:
						kernel_gemm(aOp, bOp, reference::getPointer<double>(cMem) + c_offset, reference::getPointer<double>(aMem) + a_offset,
								reference::getPointer<double>(bMem) + b_offset, reference::getAlphaValue<double>(alpha), reference::getBetaValue<double>(beta), M, N, K);
						break;
					case AVOCADO_DTYPE_COMPLEX32:
						kernel_gemm(aOp, bOp, reference::getPointer<std::complex<float>>(cMem) + c_offset, reference::getPointer<std::complex<float>>(aMem) + a_offset,
								reference::getPointer<std::complex<float>>(bMem) + b_offset, reference::getAlphaValue<std::complex<float>>(alpha),
								reference::getBetaValue<std::complex<float>>(beta), M, N, K);
						break;
					case AVOCADO_DTYPE_COMPLEX64:
						kernel_gemm(aOp, bOp, reference::getPointer<std::complex<double>>(cMem) + c_offset, reference::getPointer<std::complex<double>>(aMem) + a_offset,
								reference::getPointer<std::complex<double>>(bMem) + b_offset, reference::getAlphaValue<std::complex<double>>(alpha),
								reference::getBetaValue<std::complex<double>>(beta), M, N, K);
						break;
					default:
						return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
				}
			}
			return AVOCADO_STATUS_SUCCESS;
		}

	} /* namespace backend */
} /* namespace avocado */

