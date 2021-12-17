/*
 * reference_backend.h
 *
 *  Created on: Dec 5, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_REFERENCE_BACKEND_H_
#define AVOCADO_REFERENCE_BACKEND_H_

#include <avocado/backend/backend_defs.h>

namespace avocado
{
	namespace backend
	{
#ifdef __cplusplus
		extern "C"
		{
#endif
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
		 * \brief Creates new context.
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The context was successfully created.
		 * \retval AVOCADO_STATUS_BAD_PARAM The passed pointer is null.
		 */
		DLL_PUBLIC avStatus_t refCreateContextDescriptor(avContextDescriptor_t *result);

		/**
		 * \brief Destroys context. If null pointer is passed, the function does nothing.
		 *
		 * \param[in] context Context descriptor to be destroyed.
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The context was successfully destroyed.
		 * \retval AVOCADO_STATUS_BAD_PARAM The passed context is invalid or is a descriptor of the default context.
		 * \retval AVOCADO_STATUS_FREE_FAILED Deallocation failed.
		 */
		DLL_PUBLIC avStatus_t refDestroyContextDescriptor(avContextDescriptor_t desc);

		/**
		 * \brief Blocks until all operations in a given context are finished.
		 *
		 * \param[in] context Context descriptor to synchronize with.
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The synchronization was successfully performed.
		 * \retval AVOCADO_STATUS_BAD_PARAM The passed context is invalid.
		 */
		DLL_PUBLIC avStatus_t refSynchronizeWithContext(avContextDescriptor_t context);

		/**
		 * \brief Checks if all operations in a given context are finished.
		 *
		 * \param[in] context Context descriptor to query for readiness.
		 * \param[out] result
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The readiness was successfully checked.
		 * \retval AVOCADO_STATUS_BAD_PARAM The result pointer is null.
		 */
		DLL_PUBLIC avStatus_t refIsContextReady(avContextDescriptor_t context, bool *result);

		/**
		 * \brief Allocates new memory block and creates its descriptor.
		 *
		 * \param[out] result Pointer to new memory descriptor.
		 * \param[in] count Number of bytes to allocate.
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The memory was successfully allocated.
		 * \retval AVOCADO_STATUS_BAD_PARAM The passed pointer is null.
		 * \retval AVOCADO_STATUS_BAD_ALLOC The allocation failed.
		 */
		DLL_PUBLIC avStatus_t refCreateMemoryDescriptor(avMemoryDescriptor_t *result, avSize_t count);

		/**
		 * \brief Creates non-owning view of another memory block.
		 *
		 * \param[out] result
		 * \param[in] desc
		 * \param[in] offset
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The memory view was successfully created.
		 * \retval AVOCADO_STATUS_BAD_PARAM The descriptor is invalid or not owning or offset is negative.
		 */
		DLL_PUBLIC avStatus_t refCreateMemoryView(avMemoryDescriptor_t *result, const avMemoryDescriptor_t desc, avSize_t offset);

		/**
		 * \brief Frees memory and destroys the memory descriptor.
		 *
		 * \param[out] ptr Pointer to block of memory to be deleted. Can be null, the function does nothing then.
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The memory was successfully deleted.
		 * \retval AVOCADO_STATUS_FREE_FAILED Deallocation failed.
		 */
		DLL_PUBLIC avStatus_t refDestroyMemoryDescriptor(avMemoryDescriptor_t desc);

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
		DLL_PUBLIC avStatus_t refSetMemory(avContextDescriptor_t context, avMemoryDescriptor_t dst, avSize_t dstSize, const void *pattern,
				avSize_t patternSize);

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
		DLL_PUBLIC avStatus_t refCopyMemory(avContextDescriptor_t context, avMemoryDescriptor_t dst, const avMemoryDescriptor_t src, avSize_t count);

		/**
		 * \brief This method returns pointer associated with the memory descriptor.
		 * Because host memory and CPU device memory are actually the same, there is no method to copy the memory from CPU device to host.
		 * Instead, this method is provided to convert CPU memory descriptor into the host pointer.
		 * If anything goes wrong, a null pointer will be returned.
		 */
		DLL_PUBLIC void* refGetMemoryPointer(avMemoryDescriptor_t mem);

		/**
		 * \brief Creates new tensor descriptor.
		 */
		DLL_PUBLIC avStatus_t refCreateTensorDescriptor(avTensorDescriptor_t *result);

		/**
		 * \brief Deletes tensor descriptor.
		 */
		DLL_PUBLIC avStatus_t refDestroyTensorDescriptor(avTensorDescriptor_t desc);

		/**
		 * \brief Sets tensor descriptor.
		 */
		DLL_PUBLIC avStatus_t refSetTensorDescriptor(avTensorDescriptor_t desc, avDataType_t dtype, int nbDims, const int dimensions[]);

		/**
		 * \brief Queries parameters of tensor descriptor.
		 */
		DLL_PUBLIC avStatus_t refGetTensorDescriptor(avTensorDescriptor_t desc, avDataType_t *dtype, int *nbDims, int dimensions[]);

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
		DLL_PUBLIC avStatus_t refChangeType(avContextDescriptor_t context, avMemoryDescriptor_t dst, avDataType_t dstType,
				const avMemoryDescriptor_t src, avDataType_t srcType, avSize_t elements);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] cDesc
		 * \param[out] cMem
		 * \param[in] aDesc
		 * \param[in] aMem
		 * \param[in] nbTensors
		 */
		DLL_PUBLIC avStatus_t refConcatTensors(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem,
				const avTensorDescriptor_t aDesc[], const avMemoryDescriptor_t aMem[], int nbTensors);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] cDesc
		 * \param[out] cMem
		 * \param[in] aDesc
		 * \param[in] aMem
		 * \param[in] nbTensors
		 */
		DLL_PUBLIC avStatus_t refSplitTensors(avContextDescriptor_t context, const avTensorDescriptor_t cDesc[], avMemoryDescriptor_t cMem[],
				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, int nbTensors);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] cDesc
		 * \param[out] cMem
		 * \param[in] aDesc
		 * \param[in] aMem
		 * \param[in] newDimOrder
		 */
		DLL_PUBLIC avStatus_t refTranspose(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem,
				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const int newDimOrder[]);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] cDesc
		 * \param[out] cMem
		 * \param[in] alpha
		 */
		DLL_PUBLIC avStatus_t refScaleTensor(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem,
				const void *alpha);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] cDesc
		 * \param[out] cMem
		 * \param[in] scalar
		 */
		DLL_PUBLIC avStatus_t refAddScalarToTensor(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem,
				const void *scalar);

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
		DLL_PUBLIC avStatus_t refBinaryOp(avContextDescriptor_t context, avBinaryOp_t operation, const void *alpha1, const avTensorDescriptor_t aDesc,
				const avMemoryDescriptor_t aMem, const void *alpha2, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem,
				const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem);

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
		DLL_PUBLIC avStatus_t refUnaryOp(avContextDescriptor_t context, avUnaryOp_t operation, const void *alpha, const avTensorDescriptor_t aDesc,
				const avMemoryDescriptor_t aMem, const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem);

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
		DLL_PUBLIC avStatus_t refReduceTensor(avContextDescriptor_t context, avReduceOp_t operation, const void *alpha,
				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const void *beta, const avTensorDescriptor_t cDesc,
				avMemoryDescriptor_t cMem);

		/**
		 *
		 * C = alpha1 * activation(alpha2 * A + beta2 * C) + beta1 * C
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] alpha1
		 * \param[in] alpha2
		 * \param[in] aDesc
		 * \param[in] aMem
		 * \param[in] beta1
		 * \param[in] beta2
		 * \param[in] cDesc
		 * \param[in] cMem
		 * \param[in] activation
		 */
		DLL_PUBLIC avStatus_t refAddTensors(avContextDescriptor_t context, const void *alpha1, const void *alpha2, const avTensorDescriptor_t bDesc,
				const avMemoryDescriptor_t bMem, const void *beta1, const void *beta2, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem,
				avActivationType_t activation);

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
		DLL_PUBLIC avStatus_t refGemm(avContextDescriptor_t context, avGemmOperation_t aOp, avGemmOperation_t bOp, const void *alpha,
				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem,
				const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem);

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
		DLL_PUBLIC avStatus_t refGemmBatched(avContextDescriptor_t context, avGemmOperation_t aOp, avGemmOperation_t bOp, const void *alpha,
				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem,
				const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem);

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
		DLL_PUBLIC avStatus_t refActivationForward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem);

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
		DLL_PUBLIC avStatus_t refActivationBackward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
				const avTensorDescriptor_t yDesc, const avMemoryDescriptor_t yMem, const avTensorDescriptor_t dyDesc,
				const avMemoryDescriptor_t dyMem, const void *beta, const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem);

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
		DLL_PUBLIC avStatus_t refSoftmaxForward(avContextDescriptor_t context, avSoftmaxMode_t mode, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem);

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
		DLL_PUBLIC avStatus_t refSoftmaxBackward(avContextDescriptor_t context, avSoftmaxMode_t mode, const void *alpha,
				const avTensorDescriptor_t yDesc, const avMemoryDescriptor_t yMem, const avTensorDescriptor_t dyDesc,
				const avMemoryDescriptor_t dyMem, const void *beta, const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem);

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
		DLL_PUBLIC avStatus_t refAffineForward(avContextDescriptor_t context, avActivationType_t activation, const avTensorDescriptor_t wDesc,
				const avMemoryDescriptor_t wMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem);

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
		DLL_PUBLIC avStatus_t refBatchNormInference(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem, const avTensorDescriptor_t scaleBiasMeanVarDesc, const avMemoryDescriptor_t scaleMem,
				const avMemoryDescriptor_t biasMem, const avMemoryDescriptor_t meanMem, const avMemoryDescriptor_t varianceMem, double epsilon);

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
		DLL_PUBLIC avStatus_t refBatchNormForward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem, const avTensorDescriptor_t scaleBiasMeanVarDesc, const avMemoryDescriptor_t scaleMem,
				const avMemoryDescriptor_t biasMem, avMemoryDescriptor_t meanMem, avMemoryDescriptor_t varianceMem, double epsilon);

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
		 * \param[in] epsilon
		 */
		DLL_PUBLIC avStatus_t refBatchNormBackward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t yDesc, const avMemoryDescriptor_t yMem,
				const void *beta, const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem, const avTensorDescriptor_t dyDesc,
				avMemoryDescriptor_t dyMem, const avTensorDescriptor_t scaleMeanVarDesc, const avMemoryDescriptor_t scaleMem,
				const avMemoryDescriptor_t meanMem, const avMemoryDescriptor_t varianceMem, double epsilon);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] alpha
		 * \param[in] xDesc
		 * \param[in] xMem
		 * \param[in] dyDesc
		 * \param[in] dyMem
		 * \param[in] beta
		 * \param[in] scaleBiasDesc
		 * \param[out] scaleUpdateMem
		 * \param[out] biasUpdateMem
		 * \param[in] meanMem
		 * \param[in] varianceMem
		 * \param[in] epsilon
		 */
		DLL_PUBLIC avStatus_t refBatchNormUpdate(avContextDescriptor_t context, const void *alpha, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const avTensorDescriptor_t dyDesc, const avMemoryDescriptor_t dyMem, const void *beta,
				const avTensorDescriptor_t scaleBiasDesc, avMemoryDescriptor_t scaleUpdateMem, avMemoryDescriptor_t biasUpdateMem,
				const avMemoryDescriptor_t meanMem, const avMemoryDescriptor_t varianceMem, double epsilon);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] config
		 * \param[in] xDesc
		 * \param[in] xMem
		 * \param[in] yDesc
		 * \param[out] yMem
		 * \param[out] states
		 */
		DLL_PUBLIC avStatus_t refDropoutForward(avContextDescriptor_t context, const avDropoutDescriptor_t config, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem, avMemoryDescriptor_t states);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] config
		 * \param[in] dyDesc
		 * \param[in] dyMem
		 * \param[in] dxDesc
		 * \param[out] dxMem
		 * \param[in] states
		 */
		DLL_PUBLIC avStatus_t refDropoutBackward(avContextDescriptor_t context, const avDropoutDescriptor_t config, const avTensorDescriptor_t dyDesc,
				const avMemoryDescriptor_t dyMem, const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem, const avTensorDescriptor_t states);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] alpha
		 * \param[in] xDesc
		 * \param[in] xMem
		 * \param[in] beta
		 * \param[in] yDesc
		 * \param[out] yMem
		 */
		DLL_PUBLIC avStatus_t refPoolingForward(avContextDescriptor_t context, const avPoolingDescriptor_t config, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem);

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
		DLL_PUBLIC avStatus_t refPoolingBackward(avContextDescriptor_t context, const avPoolingDescriptor_t config, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t dyDesc,
				const avMemoryDescriptor_t dyMem, const void *beta, const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem);

		/**
		 * \brief Creates new convolution descriptor.
		 *
		 * \param[out] result
		 */
		DLL_PUBLIC avStatus_t refCreateConvolutionDescriptor(avConvolutionDescriptor_t *result);

		/**
		 * \brief Deletes convolution descriptor.
		 *
		 * \param[in] desc
		 */
		DLL_PUBLIC avStatus_t refDestroyConvolutionDescriptor(avConvolutionDescriptor_t desc);

		/**
		 * \brief Sets convolution descriptor.
		 *
		 * \param[in] desc
		 * \param[in] mode
		 * \param[in] nbDims Dimensionality of the convolution. Its value must be 1, 2 or 3.
		 * \param[in] filterSize Array with dimensions of the filter. This parameter is mandatory.
		 * \param[in] padding Array with padding offsets. This parameter is optional (can be null), a value of 0 will be used for all dimensions.
		 * \param[in] strides Array with strides. This parameter is optional (can be null), a value of 1 will be used for all dimensions.
		 * \param[in] dilation Array with dilation factors. This parameter is optional (can be null), a value of 1 will be used for all dimensions.
		 * \param[in] groups Number of groups in the convolution. Must be greaten than 0.
		 * \param[in] paddingValue Pointer to at least 16 bytes of memory with the value of tensor padding. This parameter is optional (can be null), a value of 0 will be used then.
		 */
		DLL_PUBLIC avStatus_t refSetConvolutionDescriptor(avConvolutionDescriptor_t desc, avConvolutionMode_t mode, int nbDims, const int padding[],
				const int strides[], const int dilation[], int groups, const void *paddingValue);

		/**
		 * \brief Queries parameters of convolution descriptor.
		 *
		 * \param[in] desc
		 * \param[out] mode
		 * \param[out] nbDims
		 * \param[out] filterSize
		 * \param[out] padding
		 * \param[out] strides
		 * \param[out] dilation
		 * \param[out] groups
		 * \param[out] paddingValue Pointer to at least 16 bytes of memory with the value of tensor padding. This parameter is optional (can be null), will be ignored then.
		 */
		DLL_PUBLIC avStatus_t refGetConvolutionDescriptor(avConvolutionDescriptor_t desc, avConvolutionMode_t *mode, int *nbDims, int padding[],
				int strides[], int dilation[], int *groups, void *paddingValue);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] config
		 * \param[in] filterDesc
		 * \param[in] srcDesc
		 * \param[in] srcMem
		 * \param[in] colDesc
		 * \param[out] colMem
		 */
		DLL_PUBLIC avStatus_t refIm2Col(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t filterDesc,
				const avTensorDescriptor_t srcDesc, const avMemoryDescriptor_t srcMem, const avTensorDescriptor_t colDesc,
				avMemoryDescriptor_t colMem);

		/**
		 * y = activation(alpha1 * convolve(x, w) + alpha2 * z + b) + beta * y
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] config
		 * \param[in] alpha1
		 * \param[in] xDesc
		 * \param[in] xMem
		 * \param[in] wDesc
		 * \param[in] wMem
		 * \param[in] bDesc
		 * \param[in] bMem
		 * \param[in] activation
		 * \param[in] alpha2
		 * \param[in] zDesc
		 * \param[in] zMem
		 * \param[in] beta
		 * \param[in] yDesc
		 * \param[out] yMem
		 */
		DLL_PUBLIC avStatus_t refConvolutionBiasActivationForward(avContextDescriptor_t context, const avConvolutionDescriptor_t config,
				const void *alpha1, const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t wDesc,
				const avMemoryDescriptor_t wMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha2,
				const avTensorDescriptor_t zDesc, const avMemoryDescriptor_t zMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem, const avActivationType_t activation);

		/**
		 * y = alpha * convolve(x, w) + beta * y
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
		 */
		DLL_PUBLIC avStatus_t refConvolutionForward(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
				const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem);

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
		 */
		DLL_PUBLIC avStatus_t refConvolutionUpdate(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t dyDesc,
				const avMemoryDescriptor_t dyMem, const void *beta, const avTensorDescriptor_t dwDesc, avMemoryDescriptor_t dwMem);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] config
		 * \param[in] tileSize
		 * \param[in] aDesc
		 * \param[in] aMem
		 * \param[in] cDesc
		 * \param[out] cMem
		 */
		DLL_PUBLIC avStatus_t refWinogradWeightTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, int tileSize,
				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem);

		/**
		 *  \param[in] context Context in which the operation is performed.
		 *  \param[in] config
		 *  \param[in] tileSize
		 *  \param[in] aDesc
		 *  \param[in] aMem
		 *  \param[in] cDesc
		 *  \param[out] cMem
		 */
		DLL_PUBLIC avStatus_t refWinogradInputTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, int tileSize,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem);

		/**
		 * \param[in] context Context in which the operation is performed.
		 */
		DLL_PUBLIC avStatus_t refWinogradOutputTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, int tileSize,
				const void *alpha, const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const void *beta,
				const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem, const avTensorDescriptor_t biasDesc, const avMemoryDescriptor_t biasMem,
				avActivationType_t activation);

		/**
		 * \param[in] context Context in which the operation is performed.
		 */
		DLL_PUBLIC avStatus_t refWinogradGradientTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, int tileSize,
				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem);

		/**
		 * \param[in] context Context in which the operation is performed.
		 */
		DLL_PUBLIC avStatus_t refWinogradUpdateTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, int tileSize,
				const void *alpha, const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const void *beta,
				const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem);

		/**
		 * \brief Computes chosen metric function, averaged over entire batch.
		 *
		 * \param[in] context Context in which the operation is performed.
		 */
		DLL_PUBLIC avStatus_t refMetricFunction(avContextDescriptor_t context, avMetricType_t metricType, void *result,
				const avTensorDescriptor_t outputDesc, const avMemoryDescriptor_t outputMem, const avTensorDescriptor_t targetDesc,
				const avMemoryDescriptor_t targetMem);

		/**
		 * \param[in] context Context in which the operation is performed.
		 */
		DLL_PUBLIC avStatus_t refLossFunction(avContextDescriptor_t context, avLossType_t lossType, void *result,
				const avTensorDescriptor_t outputDesc, const avMemoryDescriptor_t outputMem, const avTensorDescriptor_t targetDesc,
				const avMemoryDescriptor_t targetMem);

		/**
		 * \param[in] context Context in which the operation is performed.
		 */
		DLL_PUBLIC avStatus_t refLossGradient(avContextDescriptor_t context, avLossType_t lossType, const void *alpha, const void *beta,
				const avTensorDescriptor_t gradientDesc, avMemoryDescriptor_t gradientMem, const avTensorDescriptor_t outputDesc,
				const avMemoryDescriptor_t outputMem, const avTensorDescriptor_t targetDesc, const avMemoryDescriptor_t targetMem, bool isFused);

		/**
		 * \brief Creates new optimizer descriptor.
		 *
		 * \param[out] result
		 */
		DLL_PUBLIC avStatus_t refCreateOptimizerDescriptor(avOptimizerDescriptor_t *result);

		/**
		 * \brief Deletes optimizer descriptor.
		 *
		 * \param[in] desc
		 */
		DLL_PUBLIC avStatus_t refDestroyOptimizerDescriptor(avOptimizerDescriptor_t desc);

		/**
		 * \brief Sets optimizer descriptor.
		 *
		 * \param[in] desc
		 * \param[in] learningRate
		 * \param[in] useMomentum
		 * \param[in] useNesterov
		 * \param[in] beta1
		 */
		DLL_PUBLIC avStatus_t refSetOptimizerSGD(avOptimizerDescriptor_t desc, double learningRate, bool useMomentum, bool useNesterov, double beta1);

		/**
		 * \brief Queries parameters of optimizer descriptor.
		 *
		 * \param[in] desc
		 * \param[out] learningRate
		 * \param[out] useMomentum
		 * \param[out] useNesterov
		 * \param[out] beta1
		 */
		DLL_PUBLIC avStatus_t gefGetOptimizerSGD(avOptimizerDescriptor_t desc, double *learningRate, bool *useMomentum, bool *useNesterov,
				double *beta1);

		/**
		 * \brief Sets optimizer descriptor.
		 *
		 * \param[in] desc
		 * \param[in] learningRate
		 * \param[in] beta1
		 * \param[in] beta2
		 */
		DLL_PUBLIC avStatus_t refSetOptimizerADAM(avOptimizerDescriptor_t desc, double learningRate, double beta1, double beta2);

		/**
		 * \brief Queries parameters of optimizer descriptor.
		 *
		 * \param[in] desc
		 * \param[out] learningRate
		 * \param[out] beta1
		 * \param[out] beta2
		 */
		DLL_PUBLIC avStatus_t gefGetOptimizerADAM(avOptimizerDescriptor_t desc, double *learningRate, double *beta1, double *beta2);

		/**
		 * \brief Queries type of optimizer descriptor.
		 *
		 * \param[in] desc
		 * \param[out] type
		 */
		DLL_PUBLIC avStatus_t gefGetOptimizerType(avOptimizerDescriptor_t desc, avOptimizerType_t *type);

		/**
		 * \param[in] context Context in which the operation is performed.
		 */
		DLL_PUBLIC avStatus_t refOptimizerLearn(avContextDescriptor_t context, const avOptimizerDescriptor_t optimizer,
				const avTensorDescriptor_t weightDesc, avMemoryDescriptor_t weightMem, const avTensorDescriptor_t updateDesc,
				const avTensorDescriptor_t updateMem, avMemoryDescriptor_t workspace1, avMemoryDescriptor_t workspace2);

		/**
		 * \param[in] context Context in which the operation is performed.
		 */
		DLL_PUBLIC avStatus_t refRegularizerL2(avContextDescriptor_t context, const avTensorDescriptor_t gradientDesc,
				avMemoryDescriptor_t gradientMem, const avTensorDescriptor_t weightDesc, const avMemoryDescriptor_t weightMem,
				const void *coefficient, const void *offset, void *loss);

#ifdef __cplusplus
		}
#endif
	} /* namespace backend */
} /* namespace avocado */

#endif /* AVOCADO_REFERENCE_BACKEND_H_ */
