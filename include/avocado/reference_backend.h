/*
 * reference_backend.h
 *
 *  Created on: Nov 11, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_REFERENCE_BACKEND_H_
#define AVOCADO_REFERENCE_BACKEND_H_

#include <avocado/backend/backend_api.h>

#include <stddef.h>
#include <inttypes.h>

namespace avocado
{
	namespace backend
	{
#ifdef __cplusplus
		extern "C"
		{
#endif
		/* --------------------------------------------------------------------
		 * Implemented in 'basic_math.cpp'.
		 * --------------------------------------------------------------------
		 */

		/**
		 * \brief This routine is used to convert between data types.
		 *
		 */
		DLL_PUBLIC avStatus_t refChangeType(avContext_t context, void *dst, avDataType_t dstType, const void *src, avDataType_t srcType,
				size_t elements);
		/**
		 *
		 */
		DLL_PUBLIC avStatus_t refConcatTensors(avContext_t context, avTensor_t dst, const avTensor_t src, size_t lastDimOffsetInBytes);
		/**
		 *
		 */
		DLL_PUBLIC avStatus_t refSplitTensors(avContext_t context, avTensor_t dst, const avTensor_t src, size_t lastDimOffsetInBytes);
		/**
		 *
		 */
		DLL_PUBLIC avStatus_t refTranspose(avContext_t context, avTensor_t dst, const avTensor_t src, const int *order);
		/**
		 *
		 */
		DLL_PUBLIC avStatus_t refScaleTensor(avContext_t context, avTensor_t dst, const avScalar_t src);
		/**
		 *
		 */
		DLL_PUBLIC avStatus_t refAddScalarToTensor(avContext_t context, avTensor_t dst, const avScalar_t src);
		/**
		 *
		 */
		DLL_PUBLIC avStatus_t refOpTensor(avContext_t context, avOpTensorOp_t operation, const avScalar_t alpha1, const avTensor_t input1,
				const avScalar_t alpha2, const avTensor_t input2, const avScalar_t beta, avTensor_t output);
		/**
		 *
		 */
		DLL_PUBLIC avStatus_t refReduceTensor(avContext_t context, avReduceTensorOp_t operation, const avScalar_t alpha, const avScalar_t beta,
				const avTensor_t input, avTensor_t output);
		/**
		 *
		 */
		DLL_PUBLIC avStatus_t refAddTensors(avContext_t context, const avScalar_t alpha, const avScalar_t beta, const avTensor_t input,
				avTensor_t output, avActivation_t activation);

		/**
		 * C = alpha * opA(A) opB(B) + beta * C
		 */
		DLL_PUBLIC avStatus_t refGemm(avContext_t context, avGemmOperation_t opA, avGemmOperation_t opB, avTensor_t C, const avTensor_t A,
				const avTensor_t B, const avScalar_t alpha, const avScalar_t beta);
		/**
		 * C = alpha * opA(A) opB(B) + beta * C
		 */
		DLL_PUBLIC avStatus_t refGemmBatched(avContext_t context, avGemmOperation_t opA, avGemmOperation_t opB, avTensor_t C, const avTensor_t A,
				const avTensor_t B, const avScalar_t alpha, const avScalar_t beta);

		/*
		 * --------------------------------------------------------------------
		 * implemented in 'activations.cpp' and 'activations.hpp'
		 * --------------------------------------------------------------------
		 */

		/** \brief This routine applies a specified neuron activation function element-wise over each input value.
		 In-place operation is allowed for this routine - input and output tensor pointers may be equal.

		 \param[in] context Pointer to a previously created context. For more information, see avContext_t.
		 \param[in] activation Activation descriptor. For more information, see ActivationDescriptor.
		 \param[in] alpha, beta Scaling factors used to blend the computation result with prior value in the output layer as follows:\n
		 dstValue = alpha * result + beta * priorDstValue
		 \param[in] input Descriptor of input tensor.
		 \param[out] output Descriptor of output tensor.

		 \retval AVOCADO_STATUS_SUCCESS The function launched successfully.
		 \retval AVOCADO_STATUS_NOT_SUPPORTED The function does not support the provided configuration.
		 \retval AVOCADO_STATUS_BAD_PARAM At least one of the following conditions are met:\n
		 The parameter mode has an invalid enumerant value.\n
		 The dimensions of the input tensor and output tensor differ.\n
		 The datatype of the input tensor and output tensor differs.
		 */
		DLL_PUBLIC avStatus_t refActivationForward(avContext_t context, const avActivation_t activation, const avScalar_t alpha,
				const avScalar_t beta, const avTensor_t input, avTensor_t output);
		/** \brief This routine calculates gradient of a specified neuron activation function.
		 In-place operation is allowed for this routine - gradientPrev and gradientNext tensor pointers may be equal.

		 \param[in] context Pointer to a previously created context. For more information, see avContext_t.
		 \param[in] activationDesc Activation descriptor. For more information, see ActivationDescriptor.
		 \param[in] alpha, beta Scaling factors used to blend the computation result with prior value in the output layer as follows:\n
		 dstValue = alpha * result + beta * priorDstValue
		 \param[in] gradientNext Descriptor of gradient tensor after the layer.
		 \param[in] output Descriptor of output tensor after the layer.
		 \param[out] gradientPrev Descriptor of gradient tensor before the layer.

		 \retval AVOCADO_STATUS_SUCCESS The function launched successfully.
		 \retval AVOCADO_STATUS_NOT_SUPPORTED The function does not support the provided configuration.
		 \retval AVOCADO_STATUS_BAD_PARAM At least one of the following conditions are met:\n
		 The parameter mode has an invalid enumerant value.\n
		 The dimensions of the input tensor and output tensor differ.\n
		 The datatype of the input tensor and output tensor differs.
		 */
		DLL_PUBLIC avStatus_t refActivationBackward(avContext_t context, const avActivation_t activation, const avScalar_t alpha,
				const avScalar_t beta, avTensor_t gradientPrev, const avTensor_t gradientNext, const avTensor_t output);
		/**
		 * \brief This routine applies softmax function.
		 In-place operation is allowed for this routine - input and output tensor pointers may be equal.

		 \param[in] context Pointer to a previously created context. For more information, see avContext_t.
		 \param[in] mode Mode indicating over which dimension the function is computed. For more information, see avSoftmaxMode_t.
		 \param[in] alpha, beta Scaling factors used to blend the computation result with prior value in the output layer as follows:\n
		 dstValue = alpha * result + beta * priorDstValue
		 \param[in] input Descriptor of input tensor.
		 \param[out] output Descriptor of output tensor.

		 \retval AVOCADO_STATUS_SUCCESS The function launched successfully.
		 \retval AVOCADO_STATUS_NOT_SUPPORTED The function does not support the provided configuration.
		 \retval AVOCADO_STATUS_BAD_PARAM At least one of the following conditions are met:\n
		 The parameter mode has an invalid enumerant value.\n
		 The dimensions of the input tensor and output tensor differ.\n
		 The datatype of the input tensor and output tensor differs.
		 */
		DLL_PUBLIC avStatus_t refSoftmaxForward(avContext_t context, avSoftmaxMode_t mode, const avScalar_t alpha, const avScalar_t beta,
				const avTensor_t input, avTensor_t output);
		/**
		 * \brief This routine calculates gradient of the softmax function.
		 In-place operation is allowed for this routine - gradientPrev and gradientNext tensor pointers may be equal.

		 \param[in] context Pointer to a previously created context. For more information, see avContext_t.
		 \param[in] mode Mode indicating over which dimension the function is computed. For more information, see avSoftmaxMode_t.
		 \param[in] alpha, beta Scaling factors used to blend the computation result with prior value in the output layer as follows:\n
		 dstValue = alpha * result + beta * priorDstValue
		 \param[in] gradientNext Descriptor of gradient tensor after the layer.
		 \param[in] output Descriptor of output tensor after the layer.
		 \param[out] gradientPrev Descriptor of gradient tensor before the layer.

		 \retval AVOCADO_STATUS_SUCCESS The function launched successfully.
		 \retval AVOCADO_STATUS_NOT_SUPPORTED The function does not support the provided configuration.
		 \retval AVOCADO_STATUS_BAD_PARAM At least one of the following conditions are met:\n
		 The parameter mode has an invalid enumerant value.\n
		 The dimensions of the input tensor and output tensor differ.\n
		 The datatype of the input tensor and output tensor differs.
		 */
		DLL_PUBLIC avStatus_t refSoftmaxBackward(avContext_t context, avSoftmaxMode_t mode, const avScalar_t alpha, const avScalar_t beta,
				avTensor_t gradientPrev, const avTensor_t gradientNext, const avTensor_t output);

		/*
		 * --------------------------------------------------------------------
		 * implemented in 'batch_norm.cpp'
		 * --------------------------------------------------------------------
		 */

		/**
		 *
		 */
		DLL_PUBLIC avStatus_t refAffineForward(avContext_t context, const avScalar_t alpha, const avScalar_t beta, const avTensor_t input,
				avTensor_t output, const avTensor_t weight, const avTensor_t bias, const avActivation_t activation);
		/**
		 *
		 */
		DLL_PUBLIC avStatus_t refBatchNormInference(avContext_t context, const avScalar_t alpha, const avScalar_t beta, const avTensor_t input,
				avTensor_t output, const avTensor_t scale, const avTensor_t bias, const avTensor_t estimatedMean, const avTensor_t estimatedVariance,
				const avScalar_t epsilon, const avActivation_t activation);
		/**
		 *
		 */
		DLL_PUBLIC avStatus_t refBatchNormForward(avContext_t context, const avScalar_t alpha, const avScalar_t beta, const avTensor_t input,
				avTensor_t output, const avTensor_t scale, const avTensor_t bias, avTensor_t savedMean, avTensor_t savedVariance,
				const avScalar_t epsilon, const avActivation_t activation);
		/**
		 *
		 */
		DLL_PUBLIC avStatus_t refBatchNormBackward(avContext_t context, const avActivation_t activation, const avScalar_t alpha,
				const avScalar_t beta, const avTensor_t input, const avTensor_t output, avTensor_t gradientPrev, avTensor_t gradientNext,
				const avTensor_t scale, const avTensor_t savedMean, const avTensor_t savedVariance, const avScalar_t epsilon);
		/**
		 *
		 */
		DLL_PUBLIC avStatus_t refBatchNormUpdate(avContext_t context, const avScalar_t alpha, const avScalar_t beta, const avTensor_t input,
				const avTensor_t gradientNext, avTensor_t scaleUpdate, avTensor_t biasUpdate, const avTensor_t savedMean,
				const avTensor_t savedVariance, const avScalar_t epsilon);

		/*
		 * --------------------------------------------------------------------
		 * implemented in 'dropout.cpp'
		 * --------------------------------------------------------------------
		 */

		/**
		 *
		 */
		DLL_PUBLIC avStatus_t refDropoutForward(avContext_t context, const avDropout_t config, const avTensor_t input, avTensor_t output,
				avTensor_t states);
		/**
		 *
		 */
		DLL_PUBLIC avStatus_t refDropoutBackward(avContext_t context, const avDropout_t config, avTensor_t gradientPrev,
				const avTensor_t gradientNext, const avTensor_t states);

		/*
		 * --------------------------------------------------------------------
		 * implemented in 'pooling.cpp'
		 * --------------------------------------------------------------------
		 */

		DLL_PUBLIC avStatus_t refPoolingForward(avContext_t context, const avPooling_t config, const avScalar_t alpha, const avScalar_t beta,
				const avTensor_t input, avTensor_t output);
		DLL_PUBLIC avStatus_t refPoolingBackward(avContext_t context, const avPooling_t config, const avScalar_t alpha, const avScalar_t beta,
				const avTensor_t input, avTensor_t gradientPrev, const avTensor_t gradientNext);

		/*
		 * --------------------------------------------------------------------
		 * implemented in 'winograd_math.cpp', 'im2col.cpp' and conv.cpp
		 * --------------------------------------------------------------------
		 */

		DLL_PUBLIC avStatus_t refIm2Col(avContext_t context, const avConvolution_t config, const avTensor_t input, avTensor_t output);

		DLL_PUBLIC avStatus_t refWinogradWeightTransform(avContext_t context, const avConvolution_t config, int tileSize, const avTensor_t weights,
				avTensor_t matrices);
		DLL_PUBLIC avStatus_t refWinogradInputTransform(avContext_t context, const avConvolution_t config, int tileSize, const avTensor_t input,
				avTensor_t matrices, const avTensor_t bias, const avActivation_t activation);
		DLL_PUBLIC avStatus_t refWinogradOutputTransform(avContext_t context, const avConvolution_t config, int tileSize, const avScalar_t alpha,
				const avScalar_t beta, const avTensor_t matrices, avTensor_t output);
		DLL_PUBLIC avStatus_t refWinogradGradientTransform(avContext_t context, const avConvolution_t config, int tileSize, const avTensor_t gradient,
				avTensor_t matrices);
		DLL_PUBLIC avStatus_t refWinogradUpdateTransform(avContext_t context, const avConvolution_t config, int tileSize, const avScalar_t alpha,
				const avScalar_t beta, const avTensor_t matrices, avTensor_t update);

		/**
		 * output = activation(alpha1 * convolve(input, weights) + alpha2 * add + bias)
		 */
		DLL_PUBLIC avStatus_t refConvolutionBiasActivationForward(avContext_t context, const avConvolution_t config, const avScalar_t alpha1, const avScalar_t beta,
				const avTensor_t input, avTensor_t output, const avTensor_t weights, const avTensor_t bias, const avActivation_t activation,
				const avScalar_t alpha2, const avTensor_t add);
		/**
		 * \brief Simplified version of the above method.
		 * output = alpha * convolve(input, weights) + beta * output
		 */
		DLL_PUBLIC avStatus_t refConvolutionForward(avContext_t context, const avConvolution_t config, const avScalar_t alpha, const avScalar_t beta,
				const avTensor_t input, avTensor_t output, const avTensor_t weights);
		DLL_PUBLIC avStatus_t refConvolutionBackward(avContext_t context, const avConvolution_t config, const avScalar_t alpha, const avScalar_t beta,
				avTensor_t gradientPrev, avTensor_t gradientNext, const avTensor_t output, const avTensor_t weights, const avActivation_t activation);
		DLL_PUBLIC avStatus_t refConvolutionUpdate(avContext_t context, const avConvolution_t config, const avScalar_t alpha, const avScalar_t beta,
				const avTensor_t input, const avTensor_t gradientNext, avTensor_t weightUpdate, avTensor_t biasUpdate);

		/*
		 * --------------------------------------------------------------------
		 * implemented in 'metrics.cpp'
		 * --------------------------------------------------------------------
		 */

		/**
		 * \brief Computes chosen metric function, averaged over entire batch.
		 */
		DLL_PUBLIC avStatus_t refMetricFunction(avContext_t context, avMetricType_t metricType, avScalar_t result, const avTensor_t output,
				const avTensor_t target);

		/*
		 * --------------------------------------------------------------------
		 * implemented in 'losses.cpp'
		 * --------------------------------------------------------------------
		 */
		/**
		 *
		 */
		DLL_PUBLIC avStatus_t refLossFunction(avContext_t context, avLossType_t lossType, avScalar_t result, const avTensor_t output,
				const avTensor_t target);
		/**
		 *
		 */
		DLL_PUBLIC avStatus_t refLossGradient(avContext_t context, avLossType_t lossType, const avScalar_t alpha, const avScalar_t beta,
				avTensor_t gradient, const avTensor_t output, const avTensor_t target, bool isFused);

		/*
		 * --------------------------------------------------------------------
		 * implemented in 'optimizers.cpp'
		 * --------------------------------------------------------------------
		 */

		/**
		 *
		 */
		DLL_PUBLIC avStatus_t refOptimizerLearn(avContext_t context, const avOptimizer_t optimizer, const avScalar_t alpha, const avScalar_t beta,
				avTensor_t weight, const avTensor_t update, avTensor_t workspace1, avTensor_t workspace2);

		/*
		 * --------------------------------------------------------------------
		 * implemented in 'regularizers.cpp'
		 * --------------------------------------------------------------------
		 */

		/**
		 *
		 */
		DLL_PUBLIC avStatus_t refRegularizerL2(avContext_t context, avTensor_t gradient, const avTensor_t weight, const avScalar_t coefficient,
				const avScalar_t offset);

#ifdef __cplusplus
		}
#endif
	} /* namespace backend */
} /* namespace avocado */

#endif /* AVOCADO_REFERENCE_BACKEND_H_ */
