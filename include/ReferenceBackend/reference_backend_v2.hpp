/*
 * reference_backend_v2.hpp
 *
 *  Created on: Feb 20, 2022
 *      Author: maciek
 */

#ifndef REFERENCEBACKEND_REFERENCE_BACKEND_V2_HPP_
#define REFERENCEBACKEND_REFERENCE_BACKEND_V2_HPP_

#include "backend_descriptors.hpp"

namespace avocado
{
	namespace backend
	{
		using avocado::backend::avStatus_t;
		using avocado::backend::avDataType_t;
		using avocado::backend::avSize_t;
		using avocado::backend::avActivationType_t;
		using avocado::backend::avBinaryOp_t;
		using avocado::backend::avUnaryOp_t;
		using avocado::backend::avReduceOp_t;
		using avocado::backend::avSoftmaxMode_t;
		using avocado::backend::avMetricType_t;
		using avocado::backend::avLossType_t;
		using avocado::backend::avGemmOperation_t;
		using avocado::backend::reference::ContextDescriptor;
		using avocado::backend::reference::TensorDescriptor;
		using avocado::backend::reference::MemoryDescriptor;
		using avocado::backend::reference::ConvolutionDescriptor;
		using avocado::backend::reference::PoolingDescriptor;
		using avocado::backend::reference::DropoutDescriptor;
		using avocado::backend::reference::OptimizerDescriptor;

		/**
		 * A few words about argument types. \n
		 * Descriptor types are passed by value, const keyword is used as a hint that object associated with the descriptor will not change within the function.
		 * All pointer and array types are assumed to be pointing to host memory.
		 *
		 * A few words about argument names. \n
		 *
		 * For functions for neural network layers there are 8 types or names: \n
		 * Argument name | Meaning
		 * ------------- | -------------
		 * x, dx         | input tensor, gradient at the input
		 * y, dy         | output tensor, gradient at the output
		 * w, dw         | weight tensor, gradient of weights
		 * b, db         | bias tensor, gradient of bias
		 * z             | another input to be somehow used by the function
		 *
		 * For other kinds of functions, letters 'a' and 'b' usually indicate inputs to the function, while letter 'c' indicates the output.
		 * Typically they followed by 'Desc' for tensor descriptors, 'Mem' for memory descriptors.
		 * Sometimes there may be more than one letter in the tensor descriptor name, like 'xyDesc'. It means that both 'x' and 'y' arguments have the same descriptor.
		 *
		 * In few functions output is named 'dst' while input is 'src'.
		 *
		 * Unless specified otherwise, all scaling factors are optional (can be null pointers) and will then behave as following:\n
		 * for alpha-like types the default value is 1.
		 * for beta-like types the default value is 0.
		 * The type for alpha and beta parameters must match the types of tensors with the exceptions for:
		 *  - all integer types - alpha and beta type must be float32. Unless specified otherwise, the integer tensor elements will be casted to float32,
		 *  scaling will be performed in float32, and then the element will be casted back to appropriate integer type.
		 *  - float16, bfloat16 - alpha and beta must be float32
		 *
		 * Context specifies the device on which the operation is performed.
		 */

		/**
		 * \brief Sets memory with given pattern of bytes.
		 *
		 * \param[in] context Context in which the operation is performed.
		 * \param[out] dst Destination memory block.
		 * \param[in] dstSize Number of bytes in the destination block.
		 * \param[in] pattern Pointer to pattern to be set. Can be null, the destination memory is zeroed then and the value patternSize argument is ignored.
		 * \param[in] patternSize Number of bytes of the pattern.
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The memory was successfully set.
		 * \retval AVOCADO_STATUS_BAD_PARAM The dstSize is not a multiple of patternSize.
		 */
		DLL_PUBLIC avStatus_t refSetMemory(const ContextDescriptor &context, MemoryDescriptor &dst, avSize_t dstOffset, avSize_t dstSize,
				const void *pattern, avSize_t patternSize);

		/**
		 * \brief Copies block of memory.
		 *
		 * \param[in] context Context in which the operation is performed.
		 * \param[out] dst Destination pointer.
		 * \param[in] src Source pointer.
		 * \param[in] count Number of bytes to copy.
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The memory was successfully copied.
		 * \retval AVOCADO_STATUS_BAD_PARAM Either dst descriptor or src descriptor is invalid.
		 */
		DLL_PUBLIC avStatus_t refCopyMemory(const ContextDescriptor &context, MemoryDescriptor &dst, avSize_t dstOffset, const MemoryDescriptor &src,
				avSize_t srcOffset, avSize_t count);

		/**
		 * \brief This method returns pointer associated with the memory descriptor.
		 * Because host memory and CPU device memory are actually the same, there is no method to copy the memory from CPU device to host.
		 * Instead, this method is provided to convert CPU memory descriptor into the host pointer.
		 * If anything goes wrong, a null pointer will be returned.
		 */
		DLL_PUBLIC void* refGetMemoryPointer(MemoryDescriptor &mem);

		/*
		 *
		 * Tensor operations.
		 *
		 */

		/**
		 * \brief This routine is used to convert between data types.
		 *
		 * \param[in] context Context in which the operation is performed.
		 * \param[out] dst
		 * \param[in] dstType
		 * \param[in] src
		 * \param[in] srcType
		 * \param[in] elements
		 *
		 */
		DLL_PUBLIC avStatus_t refChangeTypeHost(const ContextDescriptor &context, void *dst, avDataType_t dstType, const void *src,
				avDataType_t srcType, avSize_t elements);

		DLL_PUBLIC avStatus_t refChangeType(const ContextDescriptor &context, MemoryDescriptor &dst, avDataType_t dstType,
				const MemoryDescriptor &src, avDataType_t srcType, avSize_t elements);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] cDesc
		 * \param[out] cMem
		 * \param[in] aDesc
		 * \param[in] aMem
		 * \param[in] nbTensors
		 */
		DLL_PUBLIC avStatus_t refConcatTensors(const ContextDescriptor &context, const TensorDescriptor &cDesc, MemoryDescriptor &cMem,
				const std::vector<TensorDescriptor> &aDesc, const std::vector<MemoryDescriptor> &aMem, int nbTensors);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] cDesc
		 * \param[out] cMem
		 * \param[in] aDesc
		 * \param[in] aMem
		 * \param[in] nbTensors
		 */
		DLL_PUBLIC avStatus_t refSplitTensors(const ContextDescriptor &context, const std::vector<TensorDescriptor> &cDesc,
				const std::vector<MemoryDescriptor> &cMem, const TensorDescriptor &aDesc, const MemoryDescriptor &aMem, int nbTensors);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] cDesc
		 * \param[out] cMem
		 * \param[in] aDesc
		 * \param[in] aMem
		 * \param[in] newDimOrder
		 */
		DLL_PUBLIC avStatus_t refTranspose(const ContextDescriptor &context, const TensorDescriptor &cDesc, MemoryDescriptor &cMem,
				const TensorDescriptor &aDesc, const MemoryDescriptor &aMem, const std::vector<int> &newDimOrder);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] aDesc
		 * \param[in] aMem
		 * \param[in] alpha
		 * \param[in] cDesc
		 * \param[out] cMem
		 */
		DLL_PUBLIC avStatus_t refScaleTensor(const ContextDescriptor &context, const TensorDescriptor &aDesc, const MemoryDescriptor &aMem,
				const void *alpha, const TensorDescriptor &cDesc, MemoryDescriptor &cMem);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] aDesc
		 * \param[in] aMem
		 * \param[in] scalar
		 * \param[in] cDesc
		 * \param[out] cMem
		 */
		DLL_PUBLIC avStatus_t refAddScalarToTensor(const ContextDescriptor &context, const TensorDescriptor &aDesc, const MemoryDescriptor &aMem,
				const void *scalar, const TensorDescriptor &cDesc, MemoryDescriptor &cMem);

		/**
		 *
		 * y = alpha3 * activation(alpha1 * x + alpha2 * b + beta1 * z) + beta2 * z
		 *
		 * Supported data type configurations:
		 *  cdDesc dtype | aDesc dtype | bDesc dtype
		 * --------------|-------------|------------
		 *  INT8         | INT8        | FLOAT32
		 *  INT32        | INT8        | FLOAT32
		 *  FLOAT16      | FLOAT16     | FLOAT32
		 *  BFLOAT16     | BFLOAT16    | FLOAT32
		 *  FLOAT32      | FLOAT32     | FLOAT32
		 *  FLOAT64      | FLOAT64     | FLOAT64
		 *
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] alpha3
		 * \param[in] alpha1
		 * \param[in] xDesc
		 * \param[in] xMem
		 * \param[in] alpha2
		 * \param[in] bDesc
		 * \param[in] bMem
		 * \param[in] yDesc
		 * \param[out] yMem
		 * \param[in] beta1
		 * \param[in] beta2
		 * \param[in] zMem
		 * \param[in] activation
		 */
		DLL_PUBLIC avStatus_t refAddBias(const ContextDescriptor &context, const void *alpha3, const void *alpha1, const TensorDescriptor &xDesc,
				const MemoryDescriptor &xMem, const void *alpha2, const TensorDescriptor &bDesc, const MemoryDescriptor &bMem,
				const TensorDescriptor &yDesc, MemoryDescriptor &yMem, const void *beta1, const void *beta2, const MemoryDescriptor &zMem,
				avActivationType_t activation);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] operation
		 * \param[in] alpha1
		 * \param[in] aDesc
		 * \param[in] aMem
		 * \param[in] alpha2
		 * \param[in] bDesc
		 * \param[in] bMem
		 * \param[in] beta
		 * \param[in] cDesc
		 * \param[out] cMem
		 */
		DLL_PUBLIC avStatus_t refBinaryOp(const ContextDescriptor &context, avBinaryOp_t operation, const void *alpha1, const TensorDescriptor &aDesc,
				const MemoryDescriptor &aMem, const void *alpha2, const TensorDescriptor &bDesc, const MemoryDescriptor &bMem, const void *beta,
				const TensorDescriptor &cDesc, MemoryDescriptor &cMem);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] operation
		 * \param[in] alpha
		 * \param[in] aDesc
		 * \param[in] aMem
		 * \param[in] beta
		 * \param[in] cDesc
		 * \param[out] cMem
		 */
		DLL_PUBLIC avStatus_t refUnaryOp(const ContextDescriptor &context, avUnaryOp_t operation, const void *alpha, const TensorDescriptor &aDesc,
				const MemoryDescriptor &aMem, const void *beta, const TensorDescriptor &cDesc, MemoryDescriptor &cMem);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] operation
		 * \param[in] alpha
		 * \param[in] aDesc
		 * \param[in] aMem
		 * \param[in] beta
		 * \param[in] cDesc
		 * \param[out] cMem
		 */
		DLL_PUBLIC avStatus_t refReduceTensor(const ContextDescriptor &context, avReduceOp_t operation, const void *alpha,
				const TensorDescriptor &aDesc, const MemoryDescriptor &aMem, const void *beta, const TensorDescriptor &cDesc, MemoryDescriptor &cMem);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] aOp
		 * \param[in] bOp
		 * \param[in] alpha
		 * \param[in] aDesc
		 * \param[in] aMem
		 * \param[in] bDesc
		 * \param[in] bMem
		 * \param[in] beta
		 * \param[in] cDesc
		 * \param[out] cMem
		 * C = alpha * opA(A) opB(B) + beta * C
		 */
		DLL_PUBLIC avStatus_t refGemm(const ContextDescriptor &context, avGemmOperation_t aOp, avGemmOperation_t bOp, const void *alpha,
				const TensorDescriptor &aDesc, const MemoryDescriptor &aMem, const TensorDescriptor &bDesc, const MemoryDescriptor &bMem,
				const void *beta, const TensorDescriptor &cDesc, MemoryDescriptor &cMem);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] aOp
		 * \param[in] bOp
		 * \param[in] alpha
		 * \param[in] aDesc
		 * \param[in] aMem
		 * \param[in] bDesc
		 * \param[in] bMem
		 * \param[in] beta
		 * \param[in] cDesc
		 * \param[out] cMem
		 * C = alpha * opA(A) opB(B) + beta * C
		 */
		DLL_PUBLIC avStatus_t refGemmBatched(const ContextDescriptor &context, avGemmOperation_t aOp, avGemmOperation_t bOp, const void *alpha,
				const TensorDescriptor &aDesc, const MemoryDescriptor &aMem, const TensorDescriptor &bDesc, const MemoryDescriptor &bMem,
				const void *beta, const TensorDescriptor &cDesc, MemoryDescriptor &cMem);

		/*
		 *
		 * Activation functions.
		 *
		 */

		/**
		 * \brief This routine applies a specified neuron activation function element-wise over each input value.
		 * In-place operation is allowed for this routine - input and output tensor pointers may be equal.
		 *
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] activation Activation descriptor. For more information, see ActivationDescriptor.
		 * \param[in] alpha
		 * \param[in] xDesc Descriptor of input tensor.
		 * \param[in] xMem
		 * \param[in] beta
		 * \param[in] yDesc
		 * \param[out] yMem
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The function launched successfully.
		 * \retval AVOCADO_STATUS_NOT_SUPPORTED The function does not support the provided configuration.
		 * \retval AVOCADO_STATUS_BAD_PARAM At least one of the following conditions are met:\n
		 * The parameter mode has an invalid enumerant value.\n
		 * The dimensions of the input tensor and output tensor differ.\n
		 * The datatype of the input tensor and output tensor differs.
		 */
		DLL_PUBLIC avStatus_t refActivationForward(const ContextDescriptor &context, avActivationType_t activation, const void *alpha,
				const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const void *beta, const TensorDescriptor &yDesc, MemoryDescriptor &yMem);

		/**
		 * \brief This routine calculates gradient of a specified neuron activation function.
		 * In-place operation is allowed for this routine - gradientPrev and gradientNext tensor pointers may be equal.
		 *
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] activationDesc Activation descriptor. For more information, see ActivationDescriptor.
		 * \param[in] alpha, beta Scaling factors used to blend the computation result with prior value in the output layer as follows:\n
		 *  dstValue = alpha * result + beta * priorDstValue
		 * \param[in] yDesc Descriptor of output tensor after the layer.
		 * \param[in] yMem
		 * \param[in] dyDesc Descriptor of gradient tensor after the layer.
		 * \param[in] dyMem
		 * \param[in] beta
		 * \param[in] dxDesc Descriptor of gradient tensor before the layer.
		 * \param[out] dxMem
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The function launched successfully.
		 * \retval AVOCADO_STATUS_NOT_SUPPORTED The function does not support the provided configuration.
		 * \retval AVOCADO_STATUS_BAD_PARAM At least one of the following conditions are met:\n
		 * The parameter mode has an invalid enumerant value.\n
		 * The dimensions of the input tensor and output tensor differ.\n
		 * The datatype of the input tensor and output tensor differs.
		 */
		DLL_PUBLIC avStatus_t refActivationBackward(const ContextDescriptor &context, avActivationType_t activation, const void *alpha,
				const TensorDescriptor &yDesc, const MemoryDescriptor &yMem, const TensorDescriptor &dyDesc, const MemoryDescriptor &dyMem,
				const void *beta, const TensorDescriptor &dxDesc, MemoryDescriptor &dxMem);

		/**
		 * \brief This routine applies softmax function.
		 * In-place operation is allowed for this routine - input and output tensor pointers may be equal.
		 *
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] activation Activation descriptor. For more information, see ActivationDescriptor.
		 * \param[in] alpha
		 * \param[in] xDesc Descriptor of input tensor.
		 * \param[in] xMem
		 * \param[in] beta
		 * \param[in] yDesc
		 * \param[out] yMem
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The function launched successfully.
		 * \retval AVOCADO_STATUS_NOT_SUPPORTED The function does not support the provided configuration.
		 * \retval AVOCADO_STATUS_BAD_PARAM At least one of the following conditions are met:\n
		 * The parameter mode has an invalid enumerant value.\n
		 * The dimensions of the input tensor and output tensor differ.\n
		 * The datatype of the input tensor and output tensor differs.
		 */
		DLL_PUBLIC avStatus_t refSoftmaxForward(const ContextDescriptor &context, avSoftmaxMode_t mode, const void *alpha,
				const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const void *beta, const TensorDescriptor &yDesc, MemoryDescriptor &yMem);

		/**
		 * \brief This routine calculates gradient of the softmax function.
		 * In-place operation is allowed for this routine - gradientPrev and gradientNext tensor pointers may be equal.
		 *
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] activationDesc Activation descriptor. For more information, see ActivationDescriptor.
		 * \param[in] alpha, beta Scaling factors used to blend the computation result with prior value in the output layer as follows:\n
		 *  dstValue = alpha * result + beta * priorDstValue
		 * \param[in] yDesc Descriptor of output tensor after the layer.
		 * \param[in] yMem
		 * \param[in] dyDesc Descriptor of gradient tensor after the layer.
		 * \param[in] dyMem
		 * \param[in] beta
		 * \param[in] dxDesc Descriptor of gradient tensor before the layer.
		 * \param[out] dxMem
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The function launched successfully.
		 * \retval AVOCADO_STATUS_NOT_SUPPORTED The function does not support the provided configuration.
		 * \retval AVOCADO_STATUS_BAD_PARAM At least one of the following conditions are met:\n
		 * The parameter mode has an invalid enumerant value.\n
		 * The dimensions of the input tensor and output tensor differ.\n
		 * The datatype of the input tensor and output tensor differs.
		 */
		DLL_PUBLIC avStatus_t refSoftmaxBackward(const ContextDescriptor &context, avSoftmaxMode_t mode, const void *alpha,
				const TensorDescriptor &yDesc, const MemoryDescriptor &yMem, const TensorDescriptor &dyDesc, const MemoryDescriptor &dyMem,
				const void *beta, const TensorDescriptor &dxDesc, MemoryDescriptor &dxMem);

		/*
		 *
		 * Batch normalization and affine transform.
		 *
		 */

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] activation
		 * \param[in] wDesc
		 * \param[in] wMem
		 * \param[in] bDesc
		 * \param[in] bMem
		 * \param[in] alpha
		 * \param[in] xDesc
		 * \param[in] xMem
		 * \param[in] beta
		 * \param[in] yDesc
		 * \param[out] yMem
		 */
		DLL_PUBLIC avStatus_t refAffineForward(const ContextDescriptor &context, avActivationType_t activation, const TensorDescriptor &wDesc,
				const MemoryDescriptor &wMem, const TensorDescriptor &bDesc, const MemoryDescriptor &bMem, const void *alpha,
				const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const void *beta, const TensorDescriptor &yDesc, MemoryDescriptor &yMem);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] activation
		 * \param[in] alpha
		 * \param[in] xDesc
		 * \param[in] xMem
		 * \param[in] beta
		 * \param[in] yDesc
		 * \param[out] yMem
		 * \param[in] scaleBiasMeanVarDesc
		 * \param[in] scaleMem
		 * \param[in] biasMem
		 * \param[in] meanMem
		 * \param[in] varianceMem
		 * \param[in] epsilon
		 */
		DLL_PUBLIC avStatus_t refBatchNormInference(const ContextDescriptor &context, avActivationType_t activation, const void *alpha,
				const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const void *beta, const TensorDescriptor &yDesc, MemoryDescriptor &yMem,
				const TensorDescriptor &scaleBiasMeanVarDesc, const MemoryDescriptor &scaleMem, const MemoryDescriptor &biasMem,
				const MemoryDescriptor &meanMem, const MemoryDescriptor &varianceMem, double epsilon);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] activation
		 * \param[in] alpha
		 * \param[in] xDesc
		 * \param[in] xMem
		 * \param[in] beta
		 * \param[in] yDesc
		 * \param[out] yMem
		 * \param[in] scaleBiasMeanVarDesc
		 * \param[in] scaleMem
		 * \param[in] biasMem
		 * \param[in] meanMem
		 * \param[in] varianceMem
		 * \param[in] epsilon
		 */
		DLL_PUBLIC avStatus_t refBatchNormForward(const ContextDescriptor &context, avActivationType_t activation, const void *alpha,
				const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const void *beta, const TensorDescriptor &yDesc, MemoryDescriptor &yMem,
				const TensorDescriptor &scaleBiasMeanVarDesc, const MemoryDescriptor &scaleMem, const MemoryDescriptor &biasMem,
				MemoryDescriptor &meanMem, MemoryDescriptor &varianceMem, double epsilon);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] activation
		 * \param[in] alpha
		 * \param[in] xDesc
		 * \param[in] xMem
		 * \param[in] yDesc
		 * \param[in] yMem
		 * \param[in] beta
		 * \param[in] dxDesc
		 * \param[out] dxMem
		 * \param[in] dyDesc
		 * \param[out] dyMem
		 * \param[in] scaleMeanVarDesc
		 * \param[in] scaleMem
		 * \param[in] meanMem
		 * \param[in] varianceMem
		 * \param[in] alpha2
		 * \param[in] beta2
		 * \param[out] scaleUpdateMem
		 * \param[out] biasUpdateMem
		 * \param[in] epsilon
		 */
		DLL_PUBLIC avStatus_t refBatchNormBackward(const ContextDescriptor &context, avActivationType_t activation, const void *alpha,
				const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const TensorDescriptor &yDesc, const MemoryDescriptor &yMem,
				const void *beta, const TensorDescriptor &dxDesc, MemoryDescriptor &dxMem, const TensorDescriptor &dyDesc, MemoryDescriptor &dyMem,
				const TensorDescriptor &scaleMeanVarDesc, const MemoryDescriptor &scaleMem, const MemoryDescriptor &meanMem,
				const MemoryDescriptor &varianceMem, const void *alpha2, const void *beta2, MemoryDescriptor &scaleUpdateMem,
				MemoryDescriptor &biasUpdateMem, double epsilon);

		/*
		 *
		 * Dropout.
		 *
		 */

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] config
		 * \param[in] xDesc
		 * \param[in] xMem
		 * \param[in] yDesc
		 * \param[out] yMem
		 * \param[out] states
		 */
		DLL_PUBLIC avStatus_t refDropoutForward(const ContextDescriptor &context, const DropoutDescriptor &config, const TensorDescriptor &xDesc,
				const MemoryDescriptor &xMem, const TensorDescriptor &yDesc, MemoryDescriptor &yMem, MemoryDescriptor &states);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] config
		 * \param[in] dyDesc
		 * \param[in] dyMem
		 * \param[in] dxDesc
		 * \param[out] dxMem
		 * \param[in] states
		 */
		DLL_PUBLIC avStatus_t refDropoutBackward(const ContextDescriptor &context, const DropoutDescriptor &config, const TensorDescriptor &dyDesc,
				const MemoryDescriptor &dyMem, const TensorDescriptor &dxDesc, MemoryDescriptor &dxMem, const TensorDescriptor &states);

		/*
		 *
		 * Pooling.
		 *
		 */

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] alpha
		 * \param[in] xDesc
		 * \param[in] xMem
		 * \param[in] beta
		 * \param[in] yDesc
		 * \param[out] yMem
		 */
		DLL_PUBLIC avStatus_t refPoolingForward(const ContextDescriptor &context, const PoolingDescriptor &config, const void *alpha,
				const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const void *beta, const TensorDescriptor &yDesc, MemoryDescriptor &yMem);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] alpha
		 * \param[in] xDesc
		 * \param[in] xMem
		 * \param[in] dyDesc
		 * \param[in] dyMem
		 * \param[in] beta
		 * \param[in] dxDesc
		 * \param[out] dxMem
		 */
		DLL_PUBLIC avStatus_t refPoolingBackward(const ContextDescriptor &context, const PoolingDescriptor &config, const void *alpha,
				const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const TensorDescriptor &dyDesc, const MemoryDescriptor &dyMem,
				const void *beta, const TensorDescriptor &dxDesc, MemoryDescriptor &dxMem);

		/*
		 *
		 * Convolutions.
		 *
		 */

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] config
		 * \param[in] filterDesc
		 * \param[in] srcDesc
		 * \param[in] srcMem
		 * \param[in] rowDesc
		 * \param[out] rowMem
		 */
		DLL_PUBLIC avStatus_t refIm2Row(const ContextDescriptor &context, ConvolutionDescriptor &config, const TensorDescriptor &filterDesc,
				const TensorDescriptor &srcDesc, const MemoryDescriptor &srcMem, const TensorDescriptor &rowDesc, MemoryDescriptor &rowMem);

		/**
		 * \brief Calculates convolution forward pass via implicit gemm algorithm.
		 * y = activation(alpha1 * convolution(x, w) + alpha2 * z + b) + beta * y
		 *
		 * \param[in] context
		 * \param[in] config
		 * \param[in] wDesc
		 * \param[in] alpha1
		 * \param[in] matricesDesc
		 * \param[in] matricesMem
		 * \param[in] yDesc
		 * \param[ouy] yMem
		 * \param[in] bDesc
		 * \param[in] bMem
		 * \param[in] alpha2
		 * \param[in] zDesc
		 * \param[in] zMem
		 * \param[in] beta
		 * \param[in] activation
		 */
		DLL_PUBLIC avStatus_t refConvolutionImplicitGemmForward(const ContextDescriptor &context, ConvolutionDescriptor &config, const void *alpha1,
				const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const TensorDescriptor &wDesc, const MemoryDescriptor &wMem,
				const TensorDescriptor &bDesc, const MemoryDescriptor &bMem, const void *alpha2, const TensorDescriptor &zDesc,
				const MemoryDescriptor &zMem, const void *beta, const TensorDescriptor &yDesc, MemoryDescriptor &yMem, avActivationType_t activation);

		/**
		 * \brief Calculates convolution forward pass via Winograd transformations using fused algorithm.
		 * y = activation(alpha1 * convolution(x, w) + alpha2 * z + b) + beta * y
		 *
		 * \param[in] context
		 * \param[in] config
		 * \param[in] wDesc
		 * \param[in] alpha1
		 * \param[in] matricesDesc
		 * \param[in] matricesMem
		 * \param[in] yDesc
		 * \param[ouy] yMem
		 * \param[in] bDesc
		 * \param[in] bMem
		 * \param[in] alpha2
		 * \param[in] zDesc
		 * \param[in] zMem
		 * \param[in] beta
		 * \param[in] activation
		 */
		DLL_PUBLIC avStatus_t refConvolutionWinogradFusedForward(const ContextDescriptor &context, ConvolutionDescriptor &config, const void *alpha1,
				const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const TensorDescriptor &wDesc, const MemoryDescriptor &wMem,
				const TensorDescriptor &bDesc, const MemoryDescriptor &bMem, const void *alpha2, const TensorDescriptor &zDesc,
				const MemoryDescriptor &zMem, const void *beta, const TensorDescriptor &yDesc, MemoryDescriptor &yMem, avActivationType_t activation);

		/**
		 * \brief Calculates initial Winograd transform of the weight (filter) tensor.
		 *
		 * \param[in] context
		 * \param[in] config
		 * \param[in] wDesc
		 * \param[in] wMem
		 * \param[in] matricesDesc
		 * \param[out] matricesMem
		 */
		DLL_PUBLIC avStatus_t refWinogradWeightTransform(const ContextDescriptor &context, ConvolutionDescriptor &config, int transformSize,
				const TensorDescriptor &wDesc, const MemoryDescriptor &wMem, const TensorDescriptor &matricesDesc, MemoryDescriptor &matricesMem);

		/**
		 * \brief Calculates initial Winograd transform of the input tensor.
		 *
		 * \param[in] context
		 * \param[in] config
		 * \param[in] wDesc
		 * \param[in] xDesc
		 * \param[in] xMem
		 * \param[in] matricesDesc
		 * \param[out] matricesMem
		 */
		DLL_PUBLIC avStatus_t refWinogradInputTransform(const ContextDescriptor &context, ConvolutionDescriptor &config, int transformSize,
				const TensorDescriptor &wDesc, const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const TensorDescriptor &matricesDesc,
				MemoryDescriptor &matricesMem);

		/**
		 * \brief Calculates final Winograd transform of the output tensor.
		 * y = activation(alpha1 * transform(matrices) + alpha2 * z + b) + beta * y
		 *
		 * \param[in] context
		 * \param[in] config
		 * \param[in] wDesc
		 * \param[in] alpha1
		 * \param[in] matricesDesc
		 * \param[in] matricesMem
		 * \param[in] yDesc
		 * \param[ouy] yMem
		 * \param[in] bDesc
		 * \param[in] bMem
		 * \param[in] alpha2
		 * \param[in] zDesc
		 * \param[in] zMem
		 * \param[in] beta
		 * \param[in] activation
		 */
		DLL_PUBLIC avStatus_t refWinogradOutputTransform(const ContextDescriptor &context, ConvolutionDescriptor &config, int transformSize,
				const TensorDescriptor &wDesc, const void *alpha1, const TensorDescriptor &matricesDesc, const MemoryDescriptor &matricesMem,
				const TensorDescriptor &yDesc, MemoryDescriptor &yMem, const TensorDescriptor &bDesc, const MemoryDescriptor &bMem,
				const void *alpha2, const TensorDescriptor &zDesc, const MemoryDescriptor &zMem, const void *beta, avActivationType_t activation);

		/**
		 * \brief Calculates initial Winograd transform of the gradient tensor.
		 *
		 * \param[in] context
		 * \param[in] config
		 * \param[in] wDesc
		 * \param[in] dyDesc
		 * \param[in] dyMem
		 * \param[in] matricesDesc
		 * \param[out] matricesMem
		 */
		DLL_PUBLIC avStatus_t refWinogradGradientTransform(const ContextDescriptor &context, ConvolutionDescriptor &config, int transformSize,
				const TensorDescriptor &wDesc, const TensorDescriptor &dyDesc, const MemoryDescriptor &dyMem, const TensorDescriptor &matricesDesc,
				MemoryDescriptor &matricesMem);

		/**
		 * \brief Calculates final Winograd transform of the weight (filter) update tensor.
		 *
		 * \param[in] context
		 * \param[in] config
		 * \param[in] alpha
		 * \param[in] matricesDesc
		 * \param[in] matricesMem
		 * \param[in] beta
		 * \param[in] dwDescc
		 * \param[out] dwMem
		 */
		DLL_PUBLIC avStatus_t refWinogradUpdateTransform(const ContextDescriptor &context, ConvolutionDescriptor &config, int transformSize,
				const void *alpha, const TensorDescriptor &matricesDesc, const MemoryDescriptor &matricesMem, const void *beta,
				const TensorDescriptor &dwDesc, MemoryDescriptor &dwMem);

		/**
		 * \brief Simplified version of the above method.
		 * y = alpha * conv(x, w) + beta * y
		 *
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] config
		 * \param[in] alpha
		 * \param[in] xDesc
		 * \param[in] xMem
		 * \param[in] wDesc
		 * \param[in] wMem
		 * \param[in] beta
		 * \param[in] yDesc
		 * \param[out] yMem
		 * \param[in] workspaceMem Memory descriptor.
		 */
		DLL_PUBLIC avStatus_t refConvolutionForward(const ContextDescriptor &context, ConvolutionDescriptor &config, const void *alpha,
				const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const TensorDescriptor &wDesc, const MemoryDescriptor &wMem,
				const void *beta, const TensorDescriptor &yDesc, MemoryDescriptor &yMem, MemoryDescriptor &workspaceMem);

		/**
		 * \brief Simplified version of the above method.
		 * y = alpha * conv(x, w) + beta * y
		 *
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] config
		 * \param[in] alpha
		 * \param[in] dxDesc
		 * \param[out] dxMem
		 * \param[in] wDesc
		 * \param[in] wMem
		 * \param[in] beta
		 * \param[in] dyDesc
		 * \param[in] dyMem
		 * \param[in] workspaceMem Memory descriptor.
		 */
		DLL_PUBLIC avStatus_t refConvolutionBackward(const ContextDescriptor &context, ConvolutionDescriptor &config, const void *alpha,
				const TensorDescriptor &dxDesc, MemoryDescriptor &dxMem, const TensorDescriptor &wDesc, const MemoryDescriptor &wMem,
				const void *beta, const TensorDescriptor &dyDesc, const MemoryDescriptor &dyMem, MemoryDescriptor &workspaceMem);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] config
		 * \param[in] alpha
		 * \param[in] xDesc
		 * \param[in] xMem
		 * \param[in] dyDesc
		 * \param[in] dyMem
		 * \param[in] beta
		 * \param[in] dwDesc
		 * \param[out] dwMem
		 * \param[in] workspaceMem Memory descriptor.
		 */
		DLL_PUBLIC avStatus_t refConvolutionUpdate(const ContextDescriptor &context, ConvolutionDescriptor &config, const void *alpha,
				const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const TensorDescriptor &dyDesc, const MemoryDescriptor &dyMem,
				const void *beta, const TensorDescriptor &dwDesc, MemoryDescriptor &dwMem, MemoryDescriptor &workspaceMem);

		/*
		 *
		 * Losses and metrics.
		 *
		 */

		/**
		 * \brief Computes chosen metric function, averaged over entire batch.
		 *
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] metricType Type of metric function to be calculated.
		 * \param[in] outputDesc Tensor descriptor of the output.
		 * \param[in] outputMem Memory descriptor of the output.
		 * \param[in] targetDesc Tensor descriptor of the target.
		 * \param[in] targetMem Memory descriptor of the target.
		 * \param[out] result Pointer to the floating point number.
		 */
		DLL_PUBLIC avStatus_t refMetricFunction(const ContextDescriptor &context, avMetricType_t metricType, const TensorDescriptor &outputDesc,
				const MemoryDescriptor &outputMem, const TensorDescriptor &targetDesc, const MemoryDescriptor &targetMem, void *result);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] lossType
		 * \param[in] outputDesc
		 * \param[in] outputMem
		 * \param[in] targetDesc
		 * \param[in] targetMem
		 * \param[out] result
		 */
		DLL_PUBLIC avStatus_t refLossFunction(const ContextDescriptor &context, avLossType_t lossType, const TensorDescriptor &outputDesc,
				const MemoryDescriptor &outputMem, const TensorDescriptor &targetDesc, const MemoryDescriptor &targetMem, void *result);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] lossType
		 * \param[in] alpha
		 * \param[in] outputDesc
		 * \param[in] outputMem
		 * \param[in] targetDesc
		 * \param[in] targetMem
		 * \param[in] beta
		 * \param[in] gradientDesc
		 * \param[out] gradientMem
		 * \param[in] isFused
		 */
		DLL_PUBLIC avStatus_t refLossGradient(const ContextDescriptor &context, avLossType_t lossType, const void *alpha,
				const TensorDescriptor &outputDesc, const MemoryDescriptor &outputMem, const TensorDescriptor &targetDesc,
				const MemoryDescriptor &targetMem, const void *beta, const TensorDescriptor &gradientDesc, MemoryDescriptor &gradientMem,
				bool isFused);

		/*
		 *
		 * Optimizers.
		 *
		 */

		/**
		 * \brief Returns number of bytes needed for the workspace of given optimizer descriptor.
		 *
		 * \param[in] desc
		 * \param[in] wDesc
		 * \param[out] result
		 */
		DLL_PUBLIC avStatus_t refGetOptimizerWorkspaceSize(avOptimizerDescriptor_t desc, const TensorDescriptor &wDesc, avSize_t *result);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] config Optimizer descriptor.
		 * \param[in] wDesc Tensor descriptor of the parameter to be updated.
		 * \param[out] wMem Memory descriptor of the parameter to be updated.
		 * \param[in] dwDesc Tensor descriptor of the gradient.
		 * \param[in] dwMem Memory descriptor of the gradient.
		 * \param[in] workspace Memory descriptor of some persistent workspace needed by the function.
		 */
		DLL_PUBLIC avStatus_t refOptimizerLearn(const ContextDescriptor &context, const avOptimizerDescriptor_t config, const void *alpha,
				const TensorDescriptor &dwDesc, const MemoryDescriptor &dwMem, const void *beta, const TensorDescriptor &wDesc,
				MemoryDescriptor &wMem, MemoryDescriptor &workspace);

		/*
		 *
		 * Regularizers.
		 *
		 */

		/**
		 * \param[in] context Context in which the operation is performed.
		 */
		DLL_PUBLIC avStatus_t refRegularizerL2(const ContextDescriptor &context, const TensorDescriptor &dwDesc, MemoryDescriptor &dwMem,
				const TensorDescriptor &wDesc, const MemoryDescriptor &wMem, const void *coefficient, const void *offset, void *loss);

	} /* namespace backend */
} /* namespace avocado */

#endif /* REFERENCEBACKEND_REFERENCE_BACKEND_V2_HPP_ */
