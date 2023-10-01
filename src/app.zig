const builtin = @import("builtin");
const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

const glfw = @import("glfw");
const Hints = glfw.Window.Hints;

const vk = @import("vk.zig");
const vk_ctx = @import("vk_context.zig");
const VkAssert = vk_ctx.VkAssert;
const BaseDispatch = vk_ctx.BaseDispatch;
const InstanceDispatch = vk_ctx.InstanceDispatch;
const DeviceDispatch = vk_ctx.DeviceDispatch;

const width = 800;
const height = 600;
const max_frames_in_flight = 2;

const validation_layers = [_][*:0]const u8{
    "VK_LAYER_KHRONOS_validation",
};

const mac_instance_extensions = [_][*:0]const u8{
    vk.extension_info.khr_portability_enumeration.name,
};

const mac_device_extensions = [_][*:0]const u8{
    vk.extension_info.khr_portability_subset.name,
};

const device_extensions = [_][*:0]const u8{
    vk.extension_info.khr_swapchain.name,
};

const QueueFamilyIndices = struct {
    const Self = @This();

    graphics_family: ?u32 = null,
    present_family: ?u32 = null,

    pub fn isComplete(self: Self) bool {
        return self.graphics_family != null and self.present_family != null;
    }
};

const SwapChainSupportDetails = struct {
    const Self = @This();

    capabilities: vk.SurfaceCapabilitiesKHR = undefined,
    formats: ArrayList(vk.SurfaceFormatKHR) = undefined,
    present_modes: ArrayList(vk.PresentModeKHR) = undefined,

    pub fn init(
        alloc: Allocator,
    ) !Self {
        var details = SwapChainSupportDetails{
            .formats = ArrayList(vk.SurfaceFormatKHR).init(alloc),
            .present_modes = ArrayList(vk.PresentModeKHR).init(alloc),
        };

        return details;
    }

    pub fn deinit(self: *Self) void {
        self.formats.deinit();
        self.present_modes.deinit();
    }
};

const Vertex = struct {
    const binding_description: vk.VertexInputBindingDescription = .{
        .binding = 0,
        .stride = @sizeOf(Vertex),
        .input_rate = .vertex,
    };

    const attribute_descriptions: [2]vk.VertexInputAttributeDescription = .{
        .{
            .binding = 0,
            .location = 0,
            .format = .r32g32_sfloat,
            .offset = @offsetOf(Vertex, "pos"),
        },
        .{
            .binding = 0,
            .location = 1,
            .format = .r32g32b32_sfloat,
            .offset = @offsetOf(Vertex, "color"),
        },
    };

    pos: [2]f32,
    color: [3]f32,
};

const vertices = [_]Vertex{
    .{ .pos = .{ 0.0, -0.5 }, .color = .{ 1.0, 0.0, 0.0 } },
    .{ .pos = .{ 0.5, 0.5 }, .color = .{ 0.0, 1.0, 0.0 } },
    .{ .pos = .{ -0.5, 0.5 }, .color = .{ 0.0, 0.0, 1.0 } },
};

pub const App = struct {
    const Self = @This();

    allocator: Allocator,
    window: glfw.Window = undefined,

    vkb: BaseDispatch = undefined,
    vki: InstanceDispatch = undefined,
    vkd: DeviceDispatch = undefined,

    instance: vk.Instance = .null_handle,
    instance_extensions: ArrayList([*:0]const u8) = undefined,
    device_extensions: ArrayList([*:0]const u8) = undefined,
    debug_messenger: vk.DebugUtilsMessengerEXT = .null_handle,

    physical_device: vk.PhysicalDevice = .null_handle,
    device: vk.Device = .null_handle,
    graphics_queue: vk.Queue = .null_handle,
    present_queue: vk.Queue = .null_handle,
    surface: vk.SurfaceKHR = .null_handle,

    swapchain: vk.SwapchainKHR = .null_handle,
    swapchain_images: []vk.Image = undefined,
    swapchain_image_views: []vk.ImageView = undefined,
    swapchain_image_format: vk.Format = undefined,
    swapchain_extent: vk.Extent2D = undefined,
    swapchain_framebuffers: []vk.Framebuffer = undefined,

    render_pass: vk.RenderPass = .null_handle,
    pipeline_layout: vk.PipelineLayout = .null_handle,
    graphics_pipeline: vk.Pipeline = .null_handle,
    command_pool: vk.CommandPool = .null_handle,
    command_buffers: []vk.CommandBuffer = undefined,

    image_available_semaphores: []vk.Semaphore = undefined,
    render_finished_semaphores: []vk.Semaphore = undefined,
    in_flight_fences: []vk.Fence = undefined,
    current_frame: u8 = 0,
    framebuffer_resized: bool = false,

    vertex_buffer: vk.Buffer = .null_handle,
    vertex_buffer_memory: vk.DeviceMemory = .null_handle,

    pub fn init(alloc: Allocator) !App {
        var app = App{
            .allocator = alloc,
        };
        try app.initWindow();

        app.vkb = try BaseDispatch.load(@as(
            vk.PfnGetInstanceProcAddr,
            @ptrCast(&glfw.getInstanceProcAddress),
        ));
        try app.initVulkan();

        return app;
    }

    pub fn deinit(self: *Self) void {
        // Device level cleanup

        self.vkd.destroyBuffer(self.device, self.vertex_buffer, null);
        self.vkd.freeMemory(self.device, self.vertex_buffer_memory, null);

        for (0..max_frames_in_flight) |i| {
            self.vkd.destroySemaphore(self.device, self.image_available_semaphores[i], null);
            self.vkd.destroySemaphore(self.device, self.render_finished_semaphores[i], null);
            self.vkd.destroyFence(self.device, self.in_flight_fences[i], null);
        }

        self.vkd.destroyCommandPool(self.device, self.command_pool, null);
        self.cleanupSwapchain();

        self.vkd.destroyPipeline(self.device, self.graphics_pipeline, null);
        self.vkd.destroyPipelineLayout(self.device, self.pipeline_layout, null);
        self.vkd.destroyRenderPass(self.device, self.render_pass, null);

        self.vkd.destroyDevice(self.device, null);

        // Instance level cleanup
        self.vki.destroyDebugUtilsMessengerEXT(self.instance, self.debug_messenger, null);
        self.vki.destroySurfaceKHR(self.instance, self.surface, null);
        self.vki.destroyInstance(self.instance, null);

        // GLFW cleanup
        self.window.destroy();
        glfw.terminate();

        // Struct level cleanup
        self.instance_extensions.deinit();
        self.device_extensions.deinit();
        self.allocator.free(self.swapchain_images);
        self.allocator.free(self.swapchain_image_views);
        self.allocator.free(self.swapchain_framebuffers);
    }

    pub fn run(self: *Self) !void {
        while (!self.window.shouldClose()) {
            glfw.pollEvents();
            try self.drawFrame();
        }

        try self.vkd.deviceWaitIdle(self.device);
    }

    fn initWindow(self: *Self) !void {
        const success = glfw.init(.{});
        if (!success) {
            return error.FailedToInitGLFW;
        }

        self.window = glfw.Window.create(width, height, "Hello Vulkan from Zig!", null, null, .{
            .client_api = Hints.ClientAPI.no_api,
        }).?;

        if (!glfw.vulkanSupported()) {
            return error.VulkanNotSupported;
        }

        self.window.setUserPointer(self);
        self.window.setFramebufferSizeCallback(framebufferResizedCallback);
    }

    fn initVulkan(self: *Self) !void {
        try self.getRequiredExtensions();
        try self.createInstance();

        if (builtin.mode == std.builtin.OptimizeMode.Debug) {
            try self.setupDebugMessenger();
        }

        try self.createSurface();
        try self.initDeviceExtensions();
        try self.pickPhysicalDevice();
        try self.createLogicalDevice();

        try self.createSwapChain();
        try self.createImageViews();
        try self.createRenderPass();
        try self.createGraphicsPipeline();
        try self.createFramebuffers();
        try self.createCommandPool();
        try self.createVertexBuffer();
        try self.createCommandBuffers();
        try self.createSyncObjects();
    }

    fn createInstance(self: *Self) !void {
        const app_info = vk.ApplicationInfo{
            .p_application_name = "Hello Triangle",
            .application_version = vk.makeApiVersion(0, 1, 3, 0),
            .p_engine_name = "No Engine",
            .engine_version = vk.makeApiVersion(0, 1, 3, 0),
            .api_version = vk.makeApiVersion(0, 1, 3, 0),
        };

        var create_info = vk.InstanceCreateInfo{
            .p_application_info = &app_info,
            .enabled_extension_count = @intCast(self.instance_extensions.items.len),
            .pp_enabled_extension_names = self.instance_extensions.items.ptr,
            .enabled_layer_count = 0,
            .flags = .{ .enumerate_portability_bit_khr = true },
            .p_next = null,
        };

        if (builtin.mode == std.builtin.OptimizeMode.Debug) {
            const debug_create_info = createDebugMessengerCreateInfo();
            for (self.instance_extensions.items, 0..) |ext, i| {
                std.log.debug("Required Instance Extension {}: {s}", .{ i, ext });
            }

            if (!(try checkValidationLayerSupport(self))) {
                return error.LayerNotPresent;
            }

            create_info.enabled_layer_count = @intCast(validation_layers.len);
            create_info.pp_enabled_layer_names = &validation_layers;
            create_info.p_next = &debug_create_info;
        }

        self.instance = try self.vkb.createInstance(&create_info, null);
        self.vki = try InstanceDispatch.load(self.instance, self.vkb.dispatch.vkGetInstanceProcAddr);
    }

    fn createSurface(self: *Self) !void {
        var result = glfw.createWindowSurface(self.instance, self.window, null, &self.surface);
        try VkAssert.withMessage(@enumFromInt(result), "Failed to create window surface");
    }

    fn pickPhysicalDevice(self: *Self) !void {
        var device_count: u32 = undefined;
        var result = try self.vki.enumeratePhysicalDevices(self.instance, &device_count, null);
        try VkAssert.withMessage(result, "Failed to find a GPU with Vulkan support.");

        var devices = try self.allocator.alloc(vk.PhysicalDevice, device_count);
        defer self.allocator.free(devices);

        result = try self.vki.enumeratePhysicalDevices(self.instance, &device_count, devices.ptr);
        try VkAssert.withMessage(result, "Failed to find a GPU with Vulkan support.");

        for (devices) |device| {
            if (try self.isDeviceSuitable(device)) {
                self.physical_device = device;
                break;
            }
        }

        if (self.physical_device == .null_handle) {
            return error.NoSuitableGPU;
        }
    }

    fn initDeviceExtensions(self: *Self) !void {
        self.device_extensions = ArrayList([*:0]const u8).init(self.allocator);
        try self.device_extensions.appendSlice(&device_extensions);
        if (builtin.os.tag == .macos) try self.device_extensions.appendSlice(&mac_device_extensions);
    }

    fn createLogicalDevice(self: *Self) !void {
        const indices = try self.findQueueFamilies(self.physical_device);
        const queue_priority: f32 = 1.0;

        var unique_queue_families = std.AutoArrayHashMap(u32, void).init(self.allocator);
        defer unique_queue_families.deinit();

        try unique_queue_families.put(indices.graphics_family.?, {});
        try unique_queue_families.put(indices.present_family.?, {});

        var queue_create_infos = try self.allocator.alloc(vk.DeviceQueueCreateInfo, unique_queue_families.count());
        defer self.allocator.free(queue_create_infos);

        for (unique_queue_families.keys(), 0..) |queue_family, i| {
            queue_create_infos[i] = vk.DeviceQueueCreateInfo{
                .queue_family_index = queue_family,
                .queue_count = 1,
                .p_queue_priorities = @ptrCast(&queue_priority),
            };
        }

        const device_features = vk.PhysicalDeviceFeatures{};
        var create_info = vk.DeviceCreateInfo{
            .p_queue_create_infos = queue_create_infos.ptr,
            .queue_create_info_count = @intCast(queue_create_infos.len),
            .p_enabled_features = &device_features,
            .enabled_extension_count = @intCast(self.device_extensions.items.len),
            .pp_enabled_extension_names = self.device_extensions.items.ptr,
        };

        if (builtin.mode == std.builtin.OptimizeMode.Debug) {
            create_info.enabled_layer_count = @intCast(validation_layers.len);
            create_info.pp_enabled_layer_names = &validation_layers;
        } else {
            create_info.enabled_layer_count = 0;
        }

        self.device = try self.vki.createDevice(self.physical_device, &create_info, null);
        self.vkd = try DeviceDispatch.load(self.device, self.vki.dispatch.vkGetDeviceProcAddr);

        self.graphics_queue = self.vkd.getDeviceQueue(self.device, indices.graphics_family.?, 0);
        self.present_queue = self.vkd.getDeviceQueue(self.device, indices.present_family.?, 0);
    }

    fn createSwapChain(self: *Self) !void {
        const swap_chain_support = try self.querySwapChainSupport(self.physical_device);
        const surface_format = Self.chooseSwapSurfaceFormat(swap_chain_support.formats.items);
        const present_mode = Self.chooseSwapPresentMode(swap_chain_support.present_modes.items);
        const extent = self.chooseSwapExtent(swap_chain_support.capabilities);

        var image_count = swap_chain_support.capabilities.min_image_count + 1;

        if (swap_chain_support.capabilities.max_image_count > 0 and image_count > swap_chain_support.capabilities.max_image_count) {
            image_count = swap_chain_support.capabilities.max_image_count;
        }

        var create_info = vk.SwapchainCreateInfoKHR{
            .surface = self.surface,
            .min_image_count = image_count,
            .image_format = surface_format.format,
            .image_color_space = surface_format.color_space,
            .image_extent = extent,
            .image_array_layers = 1,
            .image_usage = .{
                .color_attachment_bit = true,
            },
            .image_sharing_mode = undefined,
            .pre_transform = swap_chain_support.capabilities.current_transform,
            .composite_alpha = .{ .opaque_bit_khr = true },
            .present_mode = present_mode,
            .clipped = vk.TRUE,
            .old_swapchain = .null_handle,
        };

        const indices = try self.findQueueFamilies(self.physical_device);
        const queue_family_indices = [_]u32{ indices.graphics_family.?, indices.present_family.? };

        if (indices.graphics_family != indices.present_family) {
            create_info.image_sharing_mode = .concurrent;
            create_info.queue_family_index_count = 2;
            create_info.p_queue_family_indices = &queue_family_indices;
        } else {
            create_info.image_sharing_mode = .exclusive;
            create_info.queue_family_index_count = 0;
            create_info.p_queue_family_indices = null;
        }

        self.swapchain = try self.vkd.createSwapchainKHR(self.device, &create_info, null);
        errdefer self.vkd.destroySwapchainKHR(self.device, self.swapchain, null);

        var result = try self.vkd.getSwapchainImagesKHR(self.device, self.swapchain, &image_count, null);
        try VkAssert.withMessage(result, "Failed to get swapchain images.");

        self.swapchain_images = try self.allocator.alloc(vk.Image, image_count);
        result = try self.vkd.getSwapchainImagesKHR(self.device, self.swapchain, &image_count, self.swapchain_images.ptr);

        self.swapchain_image_format = surface_format.format;
        self.swapchain_extent = extent;
    }

    fn createImageViews(self: *Self) !void {
        self.swapchain_image_views = try self.allocator.alloc(vk.ImageView, self.swapchain_images.len);

        for (self.swapchain_images, self.swapchain_image_views) |image, *image_view| {
            var create_info = vk.ImageViewCreateInfo{
                .image = image,
                .view_type = .@"2d",
                .format = self.swapchain_image_format,
                .components = .{
                    .a = .identity,
                    .r = .identity,
                    .g = .identity,
                    .b = .identity,
                },
                .subresource_range = .{
                    .aspect_mask = .{ .color_bit = true },
                    .base_mip_level = 0,
                    .level_count = 1,
                    .base_array_layer = 0,
                    .layer_count = 1,
                },
            };

            image_view.* = try self.vkd.createImageView(self.device, &create_info, null);
        }
    }

    fn createRenderPass(self: *Self) !void {
        const color_attachment = vk.AttachmentDescription{
            .format = self.swapchain_image_format,
            .samples = .{ .@"1_bit" = true },
            .load_op = .clear,
            .store_op = .store,
            .stencil_load_op = .dont_care,
            .stencil_store_op = .dont_care,
            .initial_layout = .undefined,
            .final_layout = .present_src_khr,
        };

        const color_attachment_ref = vk.AttachmentReference{
            .attachment = 0,
            .layout = .color_attachment_optimal,
        };

        const dependency = vk.SubpassDependency{
            .src_subpass = vk.SUBPASS_EXTERNAL,
            .dst_subpass = 0,
            .src_stage_mask = .{ .color_attachment_output_bit = true },
            .src_access_mask = .{},
            .dst_stage_mask = .{ .color_attachment_output_bit = true },
            .dst_access_mask = .{ .color_attachment_write_bit = true },
        };

        const subpass = vk.SubpassDescription{
            .pipeline_bind_point = .graphics,
            .color_attachment_count = 1,
            .p_color_attachments = @ptrCast(&color_attachment_ref),
        };

        const render_pass_info = vk.RenderPassCreateInfo{
            .attachment_count = 1,
            .p_attachments = @ptrCast(&color_attachment),
            .subpass_count = 1,
            .p_subpasses = @ptrCast(&subpass),
            .dependency_count = 1,
            .p_dependencies = @ptrCast(&dependency),
        };

        self.render_pass = try self.vkd.createRenderPass(self.device, &render_pass_info, null);
    }

    fn createGraphicsPipeline(self: *Self) !void {
        const vert_file align(@alignOf(u32)) = @embedFile("../shaders/vert.spv").*;
        const frag_file align(@alignOf(u32)) = @embedFile("../shaders/frag.spv").*;

        var vert_shader_module = try self.createShaderModule(&vert_file);
        defer self.vkd.destroyShaderModule(self.device, vert_shader_module, null);

        var frag_shader_module = try self.createShaderModule(&frag_file);
        defer self.vkd.destroyShaderModule(self.device, frag_shader_module, null);

        var vert_shader_stage_info = vk.PipelineShaderStageCreateInfo{
            .stage = .{ .vertex_bit = true },
            .module = vert_shader_module,
            .p_name = "main",
        };

        var frag_shader_stage_info = vk.PipelineShaderStageCreateInfo{
            .stage = .{ .fragment_bit = true },
            .module = frag_shader_module,
            .p_name = "main",
        };

        var shader_stages = [_]vk.PipelineShaderStageCreateInfo{ vert_shader_stage_info, frag_shader_stage_info };

        const binding_description = Vertex.binding_description;
        const attribute_descriptions = Vertex.attribute_descriptions;

        const vertex_input_info = vk.PipelineVertexInputStateCreateInfo{
            .vertex_binding_description_count = 1,
            .p_vertex_binding_descriptions = @ptrCast(&binding_description),
            .vertex_attribute_description_count = @intCast(attribute_descriptions.len),
            .p_vertex_attribute_descriptions = &attribute_descriptions,
        };

        const input_assembly = vk.PipelineInputAssemblyStateCreateInfo{
            .topology = .triangle_list,
            .primitive_restart_enable = vk.FALSE,
        };

        const viewport = vk.Viewport{
            .x = 0.0,
            .y = 0.0,
            .width = @floatFromInt(self.swapchain_extent.width),
            .height = @floatFromInt(self.swapchain_extent.height),
            .min_depth = 0.0,
            .max_depth = 1.0,
        };

        const scissor = vk.Rect2D{
            .offset = .{ .x = 0, .y = 0 },
            .extent = self.swapchain_extent,
        };

        const dynamic_states = [_]vk.DynamicState{ .viewport, .scissor };

        const dynamic_state = vk.PipelineDynamicStateCreateInfo{
            .dynamic_state_count = @intCast(dynamic_states.len),
            .p_dynamic_states = &dynamic_states,
        };

        const viewport_state = vk.PipelineViewportStateCreateInfo{
            .viewport_count = 1,
            .p_viewports = @ptrCast(&viewport),
            .scissor_count = 1,
            .p_scissors = @ptrCast(&scissor),
        };

        const rasterizer = vk.PipelineRasterizationStateCreateInfo{
            .depth_clamp_enable = vk.FALSE,
            .rasterizer_discard_enable = vk.FALSE,
            .polygon_mode = .fill,
            .line_width = 1.0,
            .cull_mode = .{ .back_bit = true },
            .front_face = .clockwise,
            .depth_bias_enable = vk.FALSE,
            .depth_bias_clamp = 0.0,
            .depth_bias_slope_factor = 0.0,
            .depth_bias_constant_factor = 0.0,
        };

        const multisampling = vk.PipelineMultisampleStateCreateInfo{
            .sample_shading_enable = vk.FALSE,
            .rasterization_samples = .{ .@"1_bit" = true },
            .min_sample_shading = 1.0,
            .p_sample_mask = null,
            .alpha_to_coverage_enable = vk.FALSE,
            .alpha_to_one_enable = vk.FALSE,
        };

        const color_blend_attachment = vk.PipelineColorBlendAttachmentState{
            .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
            .blend_enable = vk.FALSE,
            .src_color_blend_factor = .one,
            .dst_color_blend_factor = .zero,
            .color_blend_op = .add,
            .src_alpha_blend_factor = .one,
            .dst_alpha_blend_factor = .zero,
            .alpha_blend_op = .add,
        };

        const color_blending = vk.PipelineColorBlendStateCreateInfo{
            .logic_op_enable = vk.FALSE,
            .logic_op = .copy,
            .attachment_count = 1,
            .p_attachments = @ptrCast(&color_blend_attachment),
            .blend_constants = [4]f32{ 0.0, 0.0, 0.0, 0.0 },
        };

        const pipeline_layout_info = vk.PipelineLayoutCreateInfo{
            .set_layout_count = 0,
            .p_set_layouts = null,
            .push_constant_range_count = 0,
            .p_push_constant_ranges = null,
        };

        self.pipeline_layout = try self.vkd.createPipelineLayout(self.device, &pipeline_layout_info, null);

        const pipeline_info = vk.GraphicsPipelineCreateInfo{
            .stage_count = 2,
            .p_stages = &shader_stages,
            .p_vertex_input_state = &vertex_input_info,
            .p_input_assembly_state = &input_assembly,
            .p_viewport_state = &viewport_state,
            .p_rasterization_state = &rasterizer,
            .p_multisample_state = &multisampling,
            .p_depth_stencil_state = null,
            .p_color_blend_state = &color_blending,
            .p_dynamic_state = &dynamic_state,
            .layout = self.pipeline_layout,
            .render_pass = self.render_pass,
            .subpass = 0,
            .base_pipeline_handle = .null_handle,
            .base_pipeline_index = -1,
        };

        const result = try self.vkd.createGraphicsPipelines(
            self.device,
            .null_handle,
            1,
            @ptrCast(&pipeline_info),
            null,
            @ptrCast(&self.graphics_pipeline),
        );
        try VkAssert.withMessage(result, "Failed to create graphics pipeline.");
    }

    fn createFramebuffers(self: *Self) !void {
        self.swapchain_framebuffers = try self.allocator.alloc(vk.Framebuffer, self.swapchain_image_views.len);
        errdefer self.allocator.free(self.swapchain_framebuffers);

        for (self.swapchain_image_views, 0..) |image_view, i| {
            const attachments = [_]vk.ImageView{
                image_view,
            };

            const framebuffer_info = vk.FramebufferCreateInfo{
                .render_pass = self.render_pass,
                .attachment_count = 1,
                .p_attachments = &attachments,
                .width = self.swapchain_extent.width,
                .height = self.swapchain_extent.height,
                .layers = 1,
            };

            self.swapchain_framebuffers[i] = try self.vkd.createFramebuffer(self.device, &framebuffer_info, null);
        }
    }

    fn createCommandPool(self: *Self) !void {
        const queue_family_indices = try self.findQueueFamilies(self.physical_device);

        const pool_info = vk.CommandPoolCreateInfo{
            .flags = .{ .reset_command_buffer_bit = true },
            .queue_family_index = queue_family_indices.graphics_family.?,
        };

        self.command_pool = try self.vkd.createCommandPool(self.device, &pool_info, null);
    }

    fn createVertexBuffer(self: *Self) !void {
        const buffer_info = vk.BufferCreateInfo{
            .size = @sizeOf(Vertex) * vertices.len,
            .usage = .{ .vertex_buffer_bit = true },
            .sharing_mode = .exclusive,
        };

        self.vertex_buffer = try self.vkd.createBuffer(self.device, &buffer_info, null);
        errdefer self.vkd.destroyBuffer(self.device, self.vertex_buffer, null);

        const mem_requirements = self.vkd.getBufferMemoryRequirements(self.device, self.vertex_buffer);
        const alloc_info = vk.MemoryAllocateInfo{
            .allocation_size = mem_requirements.size,
            .memory_type_index = try self.findMemoryType(
                mem_requirements.memory_type_bits,
                .{
                    .host_visible_bit = true,
                    .host_coherent_bit = true,
                },
            ),
        };

        self.vertex_buffer_memory = try self.vkd.allocateMemory(self.device, &alloc_info, null);
        errdefer self.vkd.freeMemory(self.device, self.vertex_buffer_memory, null);

        try self.vkd.bindBufferMemory(self.device, self.vertex_buffer, self.vertex_buffer_memory, 0);

        const data = try self.vkd.mapMemory(self.device, self.vertex_buffer_memory, 0, buffer_info.size, .{});
        if (data) |data_location| {
            // const slice: []Vertex = @as([*]Vertex, @ptrCast(data_location))[0..buffer_info.size];
            @memcpy(@as([*]Vertex, @ptrCast(@alignCast(data_location))), &vertices);
        }
        self.vkd.unmapMemory(self.device, self.vertex_buffer_memory);
    }

    fn createCommandBuffers(self: *Self) !void {
        const buffer_count = max_frames_in_flight;
        const alloc_info = vk.CommandBufferAllocateInfo{
            .command_buffer_count = buffer_count,
            .command_pool = self.command_pool,
            .level = .primary,
        };

        self.command_buffers = try self.allocator.alloc(vk.CommandBuffer, buffer_count);
        errdefer self.allocator.free(self.command_buffers);

        try self.vkd.allocateCommandBuffers(self.device, &alloc_info, self.command_buffers.ptr);
    }

    fn createSyncObjects(self: *Self) !void {
        const semaphore_info = vk.SemaphoreCreateInfo{};
        const fence_info = vk.FenceCreateInfo{
            .flags = .{ .signaled_bit = true },
        };

        self.image_available_semaphores = try self.allocator.alloc(vk.Semaphore, max_frames_in_flight);
        errdefer self.allocator.free(self.image_available_semaphores);

        self.render_finished_semaphores = try self.allocator.alloc(vk.Semaphore, max_frames_in_flight);
        errdefer self.allocator.free(self.render_finished_semaphores);

        self.in_flight_fences = try self.allocator.alloc(vk.Fence, max_frames_in_flight);
        errdefer self.allocator.free(self.in_flight_fences);

        for (0..max_frames_in_flight) |i| {
            self.image_available_semaphores[i] = try self.vkd.createSemaphore(self.device, &semaphore_info, null);
            self.render_finished_semaphores[i] = try self.vkd.createSemaphore(self.device, &semaphore_info, null);
            self.in_flight_fences[i] = try self.vkd.createFence(self.device, &fence_info, null);
        }
    }

    fn recordCommandBuffer(self: *Self, buffer: vk.CommandBuffer, image_index: u32) !void {
        const begin_info = vk.CommandBufferBeginInfo{
            .flags = .{},
            .p_inheritance_info = null,
        };

        try self.vkd.beginCommandBuffer(buffer, &begin_info);

        const clear_color = vk.ClearColorValue{ .float_32 = [4]f32{ 0.0, 0.0, 0.0, 0.0 } };
        const renderpass_info = vk.RenderPassBeginInfo{
            .render_pass = self.render_pass,
            .framebuffer = self.swapchain_framebuffers[image_index],
            .render_area = .{
                .extent = self.swapchain_extent,
                .offset = .{ .x = 0, .y = 0 },
            },
            .clear_value_count = 1,
            .p_clear_values = @ptrCast(&clear_color),
        };

        self.vkd.cmdBeginRenderPass(buffer, &renderpass_info, .@"inline");
        self.vkd.cmdBindPipeline(buffer, .graphics, self.graphics_pipeline);

        self.vkd.cmdBindVertexBuffers(
            buffer,
            0,
            1,
            @ptrCast(&.{self.vertex_buffer}),
            @ptrCast(&.{0}),
        );

        const viewport = vk.Viewport{
            .x = 0.0,
            .y = 0.0,
            .width = @floatFromInt(self.swapchain_extent.width),
            .height = @floatFromInt(self.swapchain_extent.height),
            .min_depth = 0.0,
            .max_depth = 1.0,
        };
        self.vkd.cmdSetViewport(buffer, 0, 1, @ptrCast(&viewport));

        const scissor = vk.Rect2D{
            .offset = .{ .x = 0.0, .y = 0.0 },
            .extent = self.swapchain_extent,
        };
        self.vkd.cmdSetScissor(buffer, 0, 1, @ptrCast(&scissor));

        self.vkd.cmdDraw(buffer, @intCast(vertices.len), 1, 0, 0);

        self.vkd.cmdEndRenderPass(buffer);
        try self.vkd.endCommandBuffer(buffer);
    }

    fn cleanupSwapchain(self: *Self) void {
        for (self.swapchain_framebuffers) |framebuffer| {
            self.vkd.destroyFramebuffer(self.device, framebuffer, null);
        }

        for (self.swapchain_image_views) |image_view| {
            self.vkd.destroyImageView(self.device, image_view, null);
        }

        self.vkd.destroySwapchainKHR(self.device, self.swapchain, null);
    }

    fn recreateSwapchain(self: *Self) !void {
        var size = self.window.getFramebufferSize();
        while (size.width == 0 or size.height == 0) {
            size = self.window.getFramebufferSize();
            glfw.waitEvents();
        }

        try self.vkd.deviceWaitIdle(self.device);

        self.cleanupSwapchain();

        try self.createSwapChain();
        try self.createImageViews();
        try self.createFramebuffers();
    }

    fn drawFrame(self: *Self) !void {
        var result = try self.vkd.waitForFences(self.device, 1, @ptrCast(&self.in_flight_fences[self.current_frame]), vk.TRUE, std.math.maxInt(u64));
        try VkAssert.withMessage(result, "Failed to wait for fences.");

        const next_image = try self.vkd.acquireNextImageKHR(
            self.device,
            self.swapchain,
            std.math.maxInt(u64),
            self.image_available_semaphores[self.current_frame],
            .null_handle,
        );

        switch (next_image.result) {
            .error_out_of_date_khr => {
                try self.recreateSwapchain();
                return;
            },
            .success, .suboptimal_khr => {},
            else => return error.FailedToAcquireSwapchainImage,
        }

        try self.vkd.resetFences(self.device, 1, @ptrCast(&self.in_flight_fences[self.current_frame]));
        try self.vkd.resetCommandBuffer(self.command_buffers[self.current_frame], .{});
        try self.recordCommandBuffer(self.command_buffers[self.current_frame], next_image.image_index);

        const wait_semaphores = [_]vk.Semaphore{self.image_available_semaphores[self.current_frame]};
        const wait_stages = [_]vk.PipelineStageFlags{.{ .color_attachment_output_bit = true }};
        const signal_semaphores = [_]vk.Semaphore{self.render_finished_semaphores[self.current_frame]};

        const submit_info = vk.SubmitInfo{
            .wait_semaphore_count = @intCast(wait_semaphores.len),
            .p_wait_semaphores = &wait_semaphores,
            .p_wait_dst_stage_mask = &wait_stages,
            .command_buffer_count = 1,
            .p_command_buffers = @ptrCast(&self.command_buffers[self.current_frame]),
            .signal_semaphore_count = @intCast(signal_semaphores.len),
            .p_signal_semaphores = &signal_semaphores,
        };

        try self.vkd.queueSubmit(self.graphics_queue, 1, @ptrCast(&submit_info), self.in_flight_fences[self.current_frame]);

        const swapchains = [_]vk.SwapchainKHR{self.swapchain};
        const present_info = vk.PresentInfoKHR{
            .wait_semaphore_count = 1,
            .p_wait_semaphores = &signal_semaphores,
            .swapchain_count = 1,
            .p_swapchains = &swapchains,
            .p_image_indices = @ptrCast(&next_image.image_index),
        };

        result = try self.vkd.queuePresentKHR(self.present_queue, &present_info);
        switch (result) {
            .error_out_of_date_khr, .suboptimal_khr => {
                self.framebuffer_resized = false;
                try self.recreateSwapchain();
            },
            .success => {},
            else => return error.FailedToPresentSwapchainImage,
        }

        self.current_frame = (self.current_frame + 1) % max_frames_in_flight;
    }

    fn findMemoryType(self: *Self, type_filter: u32, properties: vk.MemoryPropertyFlags) !u32 {
        const mem_properties = self.vki.getPhysicalDeviceMemoryProperties(self.physical_device);
        for (0..mem_properties.memory_type_count) |i| {
            const mask = @as(u32, @intCast(i)) << @intCast(i);
            if (type_filter & mask == mask and
                mem_properties.memory_types[i].property_flags.contains(properties))
            {
                return @intCast(i);
            }
        }

        return error.NoValidMemoryType;
    }

    fn createShaderModule(self: *Self, code: []const u8) !vk.ShaderModule {
        var create_info = vk.ShaderModuleCreateInfo{
            .code_size = code.len,
            .p_code = @ptrCast(@alignCast(code.ptr)),
        };

        return try self.vkd.createShaderModule(self.device, &create_info, null);
    }

    fn chooseSwapExtent(self: *Self, capabilities: vk.SurfaceCapabilitiesKHR) vk.Extent2D {
        if (capabilities.current_extent.width != std.math.maxInt(u32)) {
            return capabilities.current_extent;
        }

        const size = self.window.getFramebufferSize();
        var actual_extent = vk.Extent2D{
            .width = @intCast(size.width),
            .height = @intCast(size.height),
        };

        actual_extent.width = std.math.clamp(
            actual_extent.width,
            capabilities.min_image_extent.width,
            capabilities.max_image_extent.width,
        );

        actual_extent.height = std.math.clamp(
            actual_extent.height,
            capabilities.min_image_extent.height,
            capabilities.max_image_extent.height,
        );

        return actual_extent;
    }

    fn chooseSwapPresentMode(available_present_modes: []vk.PresentModeKHR) vk.PresentModeKHR {
        for (available_present_modes) |present_mode| {
            if (present_mode == .mailbox_khr) {
                return present_mode;
            }
        }

        return .fifo_khr;
    }

    fn chooseSwapSurfaceFormat(available_formats: []vk.SurfaceFormatKHR) vk.SurfaceFormatKHR {
        for (available_formats) |available_format| {
            if (available_format.format == .b8g8r8a8_srgb and
                available_format.color_space == .srgb_nonlinear_khr)
            {
                return available_format;
            }
        }

        return available_formats[0];
    }

    fn querySwapChainSupport(self: *Self, device: vk.PhysicalDevice) !SwapChainSupportDetails {
        var details = try SwapChainSupportDetails.init(self.allocator);
        details.capabilities = try self.vki.getPhysicalDeviceSurfaceCapabilitiesKHR(device, self.surface);

        var format_count: u32 = undefined;
        var result = try self.vki.getPhysicalDeviceSurfaceFormatsKHR(device, self.surface, &format_count, null);
        try VkAssert.withMessage(result, "Failed to get physical device surface formats.");

        if (format_count > 0) {
            try details.formats.resize(format_count);
            result = try self.vki.getPhysicalDeviceSurfaceFormatsKHR(device, self.surface, &format_count, details.formats.items.ptr);
            try VkAssert.withMessage(result, "Failed to get physical device surface formats.");
        }

        var present_mode_count: u32 = undefined;
        result = try self.vki.getPhysicalDeviceSurfacePresentModesKHR(device, self.surface, &present_mode_count, null);
        try VkAssert.withMessage(result, "Failed to get physical device surface present modes.");

        if (present_mode_count > 0) {
            try details.present_modes.resize(present_mode_count);
            result = try self.vki.getPhysicalDeviceSurfacePresentModesKHR(device, self.surface, &present_mode_count, details.present_modes.items.ptr);
            try VkAssert.withMessage(result, "Failed to get physical device surface present modes.");
        }

        return details;
    }

    fn findQueueFamilies(self: *Self, physical_device: vk.PhysicalDevice) !QueueFamilyIndices {
        var indices = QueueFamilyIndices{};

        var queue_family_count: u32 = undefined;
        self.vki.getPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, null);

        var queue_families: []vk.QueueFamilyProperties = try self.allocator.alloc(vk.QueueFamilyProperties, queue_family_count);
        defer self.allocator.free(queue_families);
        self.vki.getPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.ptr);

        for (queue_families, 0..) |queue_family, i| {
            if (queue_family.queue_flags.graphics_bit) {
                indices.graphics_family = @intCast(i);
            }

            if (indices.isComplete()) {
                break;
            }

            if (try self.vki.getPhysicalDeviceSurfaceSupportKHR(physical_device, @intCast(i), self.surface) == vk.TRUE) {
                indices.present_family = @intCast(i);
            }
        }

        return indices;
    }

    fn isDeviceSuitable(self: *Self, device: vk.PhysicalDevice) !bool {
        const indices = try self.findQueueFamilies(device);
        const device_extensions_supported = try self.checkDeviceExtensionSupport(device);

        var swap_chain_adequate = false;
        if (device_extensions_supported) {
            var swap_chain_support = try self.querySwapChainSupport(device);
            defer swap_chain_support.deinit();

            swap_chain_adequate = swap_chain_support.formats.items.len > 0 and
                swap_chain_support.present_modes.items.len > 0;
        }

        return indices.isComplete() and
            device_extensions_supported and
            swap_chain_adequate;
    }

    fn getRequiredExtensions(self: *Self) !void {
        const glfw_extensions = glfw.getRequiredInstanceExtensions() orelse return blk: {
            const err = glfw.mustGetError();
            std.log.err("Failed to get required vulkan instance extensions: error={s}", .{err.description});
            break :blk error.code;
        };

        var required_extensions = ArrayList([*:0]const u8).init(self.allocator);
        try required_extensions.appendSlice(glfw_extensions);

        if (builtin.os.tag == .macos) {
            try required_extensions.appendSlice(&mac_instance_extensions);
        }

        if (builtin.mode == std.builtin.OptimizeMode.Debug) {
            try required_extensions.append(vk.extension_info.ext_debug_utils.name);

            var property_count: u32 = undefined;
            var result = try self.vkb.enumerateInstanceExtensionProperties(null, &property_count, null);
            try VkAssert.withMessage(result, "Failed to enumerate instance extension properties");

            // Create the buffer to store the supported instance extension properties
            var extension_properties = try self.allocator.alloc(vk.ExtensionProperties, property_count);
            defer self.allocator.free(extension_properties);

            result = try self.vkb.enumerateInstanceExtensionProperties(null, &property_count, extension_properties.ptr);
            try VkAssert.withMessage(result, "Failed to enumerate instance extension properties");

            for (extension_properties) |ext| {
                std.log.debug("Available Extension: {s}", .{ext.extension_name});
            }

            // Ensure that all of the required extensions are supported
            var count = required_extensions.items.len;
            for (extension_properties) |found_ext| {
                for (required_extensions.items) |required_ext| {
                    if (strEql(&found_ext.extension_name, required_ext)) {
                        count -= 1;
                        break;
                    }
                }
            }

            if (count > 0) {
                std.log.err("Failed to find {} required extensions.", .{count});
                return error.ExtensionNotPresent;
            }
        }

        self.instance_extensions = required_extensions;
    }

    fn checkDeviceExtensionSupport(self: *Self, device: vk.PhysicalDevice) !bool {
        var extension_count: u32 = 0;
        var result = try self.vki.enumerateDeviceExtensionProperties(device, null, &extension_count, null);
        try VkAssert.basic(result);

        var available_extensions = try self.allocator.alloc(vk.ExtensionProperties, extension_count);
        defer self.allocator.free(available_extensions);

        result = try self.vki.enumerateDeviceExtensionProperties(
            device,
            null,
            &extension_count,
            available_extensions.ptr,
        );
        try VkAssert.basic(result);

        outer: for (self.device_extensions.items) |required_ext| {
            for (available_extensions) |available_ext| {
                if (strEql(&available_ext.extension_name, required_ext)) {
                    continue :outer;
                }
            }

            return false;
        }

        return true;
    }

    fn checkValidationLayerSupport(self: *Self) !bool {
        var layer_count: u32 = undefined;
        var result = try self.vkb.enumerateInstanceLayerProperties(&layer_count, null);
        try VkAssert.withMessage(result, "Failed to enumerate instance layer properties.");

        var available_layers = try self.allocator.alloc(vk.LayerProperties, layer_count);
        defer self.allocator.free(available_layers);

        result = try self.vkb.enumerateInstanceLayerProperties(&layer_count, available_layers.ptr);
        try VkAssert.withMessage(result, "Failed to enumerate instance layer properties.");

        for (validation_layers) |layer_name| {
            var layer_found = false;

            for (available_layers) |layer_properties| {
                if (strEql(&layer_properties.layer_name, layer_name)) {
                    layer_found = true;
                    break;
                }
            }

            if (!layer_found) {
                return false;
            }
        }

        return true;
    }

    fn setupDebugMessenger(self: *Self) !void {
        const create_info = createDebugMessengerCreateInfo();
        self.debug_messenger = try self.vki.createDebugUtilsMessengerEXT(self.instance, &create_info, null);
    }
};

fn framebufferResizedCallback(window: glfw.Window, w_width: u32, w_height: u32) void {
    _ = w_width;
    _ = w_height;

    if (window.getUserPointer(App)) |app| {
        app.framebuffer_resized = true;
    }
}

fn createDebugMessengerCreateInfo() vk.DebugUtilsMessengerCreateInfoEXT {
    return .{
        .message_severity = .{ .verbose_bit_ext = true, .error_bit_ext = true, .warning_bit_ext = true },
        .message_type = .{ .general_bit_ext = true, .validation_bit_ext = true, .performance_bit_ext = true },
        .pfn_user_callback = &debugCallback,
    };
}

fn debugCallback(
    message_severity: vk.DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk.DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: ?*const vk.DebugUtilsMessengerCallbackDataEXT,
    p_user_data: ?*anyopaque,
) callconv(vk.vulkan_call_conv) vk.Bool32 {
    _ = p_user_data;
    _ = message_type;
    _ = message_severity;

    if (p_callback_data) |data| {
        std.log.debug("{s}", .{data.p_message});
    }

    return vk.TRUE;
}

// This is a weird hack to handle the fact that many vulkan names are pre-allocated
// arrays of length 256. This allows us to test against them with a standard C-style
// char ptr.
fn strEql(arr: []const u8, str: [*:0]const u8) bool {
    for (arr, str) |c1, c2| {
        if (c1 != c2) return false;
        if (c1 == 0 or c2 == 0) break;
    }

    return true;
}
