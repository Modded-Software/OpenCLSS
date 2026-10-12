/*
 * coopmat_peak - sustained register-only cooperative-matrix TFLOP/s at a required subgroup
 * size: one subgroup per workgroup, coopmat_peak.comp's loop, dispatched back to back for a
 * warm-up and then a timed interval (default 2 s each).
 *
 *   work/coopmat_peak work/coopmat_peak_n8.spv SUBGROUP [SECONDS] [TN]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

#define CHECK(call) do { VkResult r_ = (call); if (r_ != VK_SUCCESS) {               \
	fprintf(stderr, "%s:%d: %s: VkResult %d\n", __FILE__, __LINE__, #call, r_); \
	exit(1); } } while (0)

enum { GROUPS = 8192, ITERS = 1024, CHAINS = 8 };

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static VkBuffer make(VkDevice dev, VkPhysicalDevice pd, VkDeviceSize size, VkDeviceMemory *mem)
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
	void *p; CHECK(vkMapMemory(dev, *mem, 0, VK_WHOLE_SIZE, 0, &p));
	memset(p, 0, size);
	return b;
}

int main(int argc, char **argv)
{
	if (argc < 3) { fprintf(stderr, "usage: %s SPV SUBGROUP [SECONDS] [TN]\n", argv[0]); return 2; }
	uint32_t sg = (uint32_t)atoi(argv[2]);
	double seconds = argc > 3 ? atof(argv[3]) : 2.0;
	uint32_t tn = argc > 4 ? (uint32_t)atoi(argv[4]) : 8;
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
		.pNext = &v13, .shaderFloat16 = VK_TRUE };
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

	VkDeviceMemory mo, mi;
	VkBuffer bo = make(dev, pd, (VkDeviceSize)GROUPS * 8 * tn * 4, &mo);
	VkBuffer bi = make(dev, pd, 4096, &mi);

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
	uint32_t spec[2] = { sg, ITERS };
	VkSpecializationMapEntry ent[2] = { { 0, 0, 4 }, { 1, 4, 4 } };
	VkSpecializationInfo si_ = { 2, ent, sizeof spec, spec };
	VkPipelineShaderStageRequiredSubgroupSizeCreateInfo req = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
		.requiredSubgroupSize = sg };
	VkComputePipelineCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = sg ? &req : NULL,
			   .flags = sg ? VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT : 0,
			   .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main",
			   .pSpecializationInfo = &si_ },
		.layout = pl };
	if (!sg) spec[0] = 32;
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
	vkCmdDispatch(cb, GROUPS, 1, 1);
	CHECK(vkEndCommandBuffer(cb));
	VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };

	double flops_per = (double)GROUPS * ITERS * CHAINS * 2.0 * 8 * tn * 16;
	for (int phase = 0; phase < 2; phase++) {
		long runs = 0;
		double t0 = now();
		while (now() - t0 < seconds) {
			CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
			CHECK(vkQueueWaitIdle(queue));
			runs++;
		}
		double dt = now() - t0;
		if (phase)
			printf("%s subgroup %u tile 8x%ux16: %.2f TFLOP/s sustained over %.2f s (%ld dispatches, %.2f ms each)\n",
			       props.deviceName, sg, tn, flops_per * runs / dt / 1e12, dt, runs, 1e3 * dt / runs);
	}
	return 0;
}
