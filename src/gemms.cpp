/*
 * gemms.cpp
 *
 *  Created on: Nov 12, 2021
 *      Author: Maciej Kozarzewski
 */

#include <avocado/reference_backend.h>
#include "utils.hpp"

namespace
{
	using namespace avocado::backend;
	template<typename T, typename U>
	void kernel_gemm_AB(T *C, const U *A, const U *B, T alpha, T beta, size_t M, size_t N, size_t K)
	{
		if (beta == zero<T>())
			clear(C, M * N);
		// C (M x N) = A (M x K) * B (K x N)
		for (int m = 0; m < M; m++)
			for (int n = 0; n < N; n++)
			{
				T tmp = zero<T>();
				for (int k = 0; k < K; k++)
					tmp += static_cast<T>(A[m * K + k]) * static_cast<T>(B[k * N + n]);
				C[m * N + n] = alpha * tmp + beta * C[m * N + n];
			}
	}
	template<typename T, typename U>
	void baseline_gemm_ABT(T *C, const U *A, const U *B, T alpha, T beta, size_t M, size_t N, size_t K)
	{
		if (beta == zero<T>())
			clear(C, M * N);
		// C (M x N) = A (M x K) * B (N x K)
		for (int m = 0; m < M; m++)
			for (int n = 0; n < N; n++)
			{
				T tmp = zero<T>();
				for (int k = 0; k < K; k++)
					tmp += static_cast<T>(A[m * K + k]) * static_cast<T>(B[n * K + k]);
				C[m * N + n] = alpha * tmp + beta * C[m * N + n];
			}
	}
	template<typename T, typename U>
	void baseline_gemm_ATB(T *C, const U *A, const U *B, T alpha, T beta, size_t M, size_t N, size_t K)
	{
		if (beta == zero<T>())
			clear(C, M * N);
		// C (M x N) = A (K x N) * B (K x N)
		for (int m = 0; m < M; m++)
			for (int n = 0; n < N; n++)
			{
				T tmp = zero<T>();
				for (int k = 0; k < K; k++)
					tmp += static_cast<T>(A[k * N + n]) * static_cast<T>(B[k * N + n]);
				C[m * N + n] = alpha * tmp + beta * C[m * N + n];
			}
	}
	template<typename T, typename U>
	void baseline_gemm_ATBT(T *C, const U *A, const U *B, T alpha, T beta, size_t M, size_t N, size_t K)
	{
		if (beta == zero<T>())
			clear(C, M * N);
		// C (M x N) = A (K x M) * B (N x K)
		for (int m = 0; m < M; m++)
			for (int n = 0; n < N; n++)
			{
				T tmp = zero<T>();
				for (int k = 0; k < K; k++)
					tmp += static_cast<T>(A[k * N + n]) * static_cast<T>(B[n * K + k]);
				C[m * N + n] = alpha * tmp + beta * C[m * N + n];
			}
	}
}
namespace avocado
{
	namespace backend
	{
		avStatus_t refGemm(avContext_t context, avGemmOperation_t opA, avGemmOperation_t opB, avTensor_t C, const avTensor_t A, const avTensor_t B,
				const avScalar_t alpha, const avScalar_t beta)
		{
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refGemmBatched(avContext_t context, avGemmOperation_t opA, avGemmOperation_t opB, avTensor_t C, const avTensor_t A,
				const avTensor_t B, const avScalar_t alpha, const avScalar_t beta)
		{
			return AVOCADO_STATUS_SUCCESS;
		}

	} /* namespace backend */
} /* namespace avocado */

