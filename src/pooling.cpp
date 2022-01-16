/*
 * pooling.cpp
 *
 *  Created on: Nov 26, 2021
 *      Author: Maciej Kozarzewski
 */

#include <backend_descriptors.hpp>
#include <ReferenceBackend/reference_backend.h>

#include "utils.hpp"

namespace
{

}

namespace avocado
{
	namespace backend
	{
		DLL_PUBLIC avStatus_t refPoolingForward(avContextDescriptor_t context, const avPoolingDescriptor_t config, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem)
		{
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}
		DLL_PUBLIC avStatus_t refPoolingBackward(avContextDescriptor_t context, const avPoolingDescriptor_t config, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t dyDesc,
				const avMemoryDescriptor_t dyMem, const void *beta, const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem)
		{
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}
	} /* namespace backend */
} /* namespace avocado */

