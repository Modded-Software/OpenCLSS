/*
 * coopmat_width - runs coopmat_width.comp with the subgroup pinned at 32 (and full
 * subgroups) exactly as libxmx pins it, then reports the subgroup layout the 256-lane
 * workgroup really got and checks the fused kernels' shared-memory fragment traffic
 * against a CPU product. Inputs are small multiples of 1/8, so every sum is exact and a
 * single wrong element is a real fault, not rounding. Exit status 0 only if all is exact.
 *
 *   make work/coopmat_width work/coopmat_width.spv && work/coopmat_width work/coopmat_width.spv
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vulkan/vulkan.h>

#define CHECK(call) do { VkResult r_ = (call); if (r_ != VK_SUCCESS) {               \
	fprintf(stderr, "%s:%d: %s: VkResult %d\n", __FILE__, __LINE__, #call, r_); \
	exit(1); } } while (0)

static uint16_t to_half(float f)          /* exact for the k/8 values used here */
{
	if (f == 0.0f) return 0;
	uint32_t b; memcpy(&b, &f, 4);
	uint32_t s = (b >> 16) & 0x8000u, e = ((b >> 23) & 0xff) - 127 + 15, m = (b >> 13) & 0x3ff;
	return (uint16_t)(s | (e << 10) | m);
}

static VkBuffer make(VkDevice dev, VkPhysicalDevice pd, VkDeviceSize size, VkDeviceMemory *mem, void **map)
{
	VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
				   .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT };
	VkBuffer b; CHECK(vkCreateBuffer(dev, &bci, NULL, &b));
	VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, b, &mr);
	VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
	VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	uint32_t t = 0;
	while (!((mr.memoryTypeBits >> t) & 1) || (mp.memoryTypes[t].propertyFlags & want) != want) t++;
	VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
				     .allocationSize = mr.size, .memoryTypeIndex = t };
	CHECK(vkAllocateMemory(dev, &mai, NULL, mem));
	CHECK(vkBindBufferMemory(dev, b, *mem, 0));
	CHECK(vkMapMemory(dev, *mem, 0, VK_WHOLE_SIZE, 0, map));
	memset(*map, 0xff, size);
	return b;
}

int main(int argc, char **argv)
{
	if (argc != 2) { fprintf(stderr, "usage: %s coopmat_width.spv\n", argv[0]); return 2; }
	FILE *f = fopen(argv[1], "rb");
	if (!f) { perror(argv[1]); return 1; }
	fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
	uint32_t *code = malloc(n);
	if (fread(code, 1, n, f) != (size_t)n) return 1;
	fclose(f);

	VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3 };
	VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app };
	VkInstance inst; CHECK(vkCreateInstance(&ici, NULL, &inst));
	uint32_t np = 1; VkPhysicalDevice pd;
	vkEnumeratePhysicalDevices(inst, &np, &pd);
	VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd, &props);

	VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR, .cooperativeMatrix = VK_TRUE };
	VkPhysicalDeviceVulkan13Features v13 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
		.pNext = &cm, .subgroupSizeControl = VK_TRUE, .computeFullSubgroups = VK_TRUE };
	VkPhysicalDeviceVulkan12Features v12 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
		.pNext = &v13, .shaderFloat16 = VK_TRUE, .storageBuffer8BitAccess = VK_FALSE };
	VkPhysicalDeviceVulkan11Features v11 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
		.pNext = &v12, .storageBuffer16BitAccess = VK_TRUE };
	const char *ext[] = { VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME };
	float prio = 1.0f;
	VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
					.queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &prio };
	VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &v11,
				   .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
				   .enabledExtensionCount = 1, .ppEnabledExtensionNames = ext };
	VkDevice dev; CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
	VkQueue queue; vkGetDeviceQueue(dev, 0, 0, &queue);

	enum { IDS = 1024 * 4, QK = 4096 * 4, PV = 2048 * 4 };
	VkDeviceMemory mo, mi; void *po, *pi;
	VkBuffer bo = make(dev, pd, IDS + QK + PV, &mo, &po);
	VkBuffer bi = make(dev, pd, (2048 * 3 + 4096) * 2, &mi, &pi);
	float Q[2048], K[2048], P[4096], V[2048];
	uint16_t *h = pi;
	srand(7);
	for (int i = 0; i < 2048; i++) { Q[i] = (rand() % 9 - 4) / 8.0f; h[i] = to_half(Q[i]); }
	for (int i = 0; i < 2048; i++) { K[i] = (rand() % 9 - 4) / 8.0f; h[2048 + i] = to_half(K[i]); }
	for (int i = 0; i < 4096; i++) { P[i] = (rand() % 9 - 4) / 8.0f; h[4096 + i] = to_half(P[i]); }
	for (int i = 0; i < 2048; i++) { V[i] = (rand() % 9 - 4) / 8.0f; h[8192 + i] = to_half(V[i]); }

	VkDescriptorSetLayoutBinding bind[2] = {
		{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
		  .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
		{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
		  .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT } };
	VkDescriptorSetLayoutCreateInfo dslci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
						  .bindingCount = 2, .pBindings = bind };
	VkDescriptorSetLayout dsl; CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));
	VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 };
	VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
					    .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps };
	VkDescriptorPool dp; CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
	VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
					     .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl };
	VkDescriptorSet set; CHECK(vkAllocateDescriptorSets(dev, &dsai, &set));
	VkDescriptorBufferInfo dbi[2] = { { bo, 0, VK_WHOLE_SIZE }, { bi, 0, VK_WHOLE_SIZE } };
	VkWriteDescriptorSet w[2];
	for (int i = 0; i < 2; i++)
		w[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
			.dstBinding = i, .descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi[i] };
	vkUpdateDescriptorSets(dev, 2, w, 0, NULL);
	VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
					    .setLayoutCount = 1, .pSetLayouts = &dsl };
	VkPipelineLayout pl; CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
	VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
					  .codeSize = n, .pCode = code };
	VkShaderModule sm; CHECK(vkCreateShaderModule(dev, &smci, NULL, &sm));
	VkPipelineShaderStageRequiredSubgroupSizeCreateInfo req = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
		.requiredSubgroupSize = 32 };
	VkComputePipelineCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = &req,
			   .flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT,
			   .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main" },
		.layout = pl };
	VkPipeline pipe; CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));

	VkCommandPoolCreateInfo cpi = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = 0 };
	VkCommandPool cp; CHECK(vkCreateCommandPool(dev, &cpi, NULL, &cp));
	VkCommandBufferAllocateInfo cbai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
					     .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
					     .commandBufferCount = 1 };
	VkCommandBuffer cb; CHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));
	VkCommandBufferBeginInfo bbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	CHECK(vkBeginCommandBuffer(cb, &bbi));
	vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
	vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0, NULL);
	vkCmdDispatch(cb, 1, 1, 1);
	CHECK(vkEndCommandBuffer(cb));
	VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
	CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
	CHECK(vkQueueWaitIdle(queue));

	uint32_t *ids = po;
	float *qk = (float *)((char *)po + IDS), *pv = (float *)((char *)po + IDS + QK);
	printf("%s\n", props.deviceName);
	int bad_ids = 0;
	for (int l = 0; l < 256; l++) {
		uint32_t sz = ids[l * 4], sg = ids[l * 4 + 1], in = ids[l * 4 + 2], ns = ids[l * 4 + 3];
		if (l < 2 || l == 31 || l == 32 || l == 255)
			printf("  lane %3d: subgroupSize %u subgroupID %u invocation %u numSubgroups %u\n",
			       l, sz, sg, in, ns);
		if (sz != 32 || sg != (uint32_t)l / 32 || in != (uint32_t)l % 32 || ns != 8) bad_ids++;
	}
	printf("ids: %d of 256 lanes not the 32-wide layout\n", bad_ids);
	int bad_qk = 0, bad_pv = 0, first = -1;
	for (int r = 0; r < 64; r++)
		for (int c = 0; c < 64; c++) {
			float s = 0;
			for (int x = 0; x < 32; x++) s += Q[r * 32 + x] * K[c * 32 + x];
			if (qk[r * 64 + c] != s) { if (first < 0) first = r * 64 + c; bad_qk++; }
		}
	if (first >= 0) printf("qk first bad at row %d col %d: %g vs %g\n", first / 64, first % 64,
			       qk[first], 0.0);
	for (int r = 0; r < 64; r++)
		for (int c = 0; c < 32; c++) {
			float s = 0;
			for (int x = 0; x < 64; x++) s += P[r * 64 + x] * V[x * 32 + c];
			if (pv[r * 32 + c] != s) bad_pv++;
		}
	printf("qk: %d of 4096 wrong; pv: %d of 2048 wrong\n", bad_qk, bad_pv);
	return bad_ids || bad_qk || bad_pv;
}
