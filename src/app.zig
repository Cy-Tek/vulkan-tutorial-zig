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

const width = 800;
const height = 600;

const validation_layers = [_][*:0]const u8{
    "VK_LAYER_KHRONOS_validation",
};

pub const App = struct {
    const Self = @This();

    allocator: Allocator,
    window: glfw.Window,

    vkb: BaseDispatch = undefined,
    vki: InstanceDispatch = undefined,

    instance: vk.Instance = undefined,
    extensions: ArrayList([*:0]const u8) = undefined,
    debug_messenger: vk.DebugUtilsMessengerEXT = undefined,
    physical_device: vk.PhysicalDevice = .null_handle,

    pub fn init(alloc: Allocator) !App {
        var window = try initWindow();
        errdefer deinitGlfw(&window);

        const base_dispatch = try BaseDispatch.load(@as(
            vk.PfnGetInstanceProcAddr,
            @ptrCast(&glfw.getInstanceProcAddress),
        ));

        var app = App{
            .allocator = alloc,
            .window = window,
            .vkb = base_dispatch,
        };

        try app.initVulkan();

        return app;
    }

    pub fn deinit(self: *Self) void {
        self.vki.destroyDebugUtilsMessengerEXT(self.instance, self.debug_messenger, null);
        self.vki.destroyInstance(self.instance, null);
        self.extensions.deinit();

        deinitGlfw(&self.window);
    }

    pub fn run(self: *Self) !void {
        while (!self.window.shouldClose()) {
            glfw.pollEvents();
        }
    }

    fn initWindow() !glfw.Window {
        _ = glfw.init(.{});
        return glfw.Window.create(width, height, "Hello Vulkan from Zig!", null, null, .{
            .client_api = Hints.ClientAPI.no_api,
            .resizable = false,
        }).?;
    }

    fn initVulkan(self: *Self) !void {
        try self.getRequiredExtensions();
        try self.createInstance();

        if (builtin.mode == std.builtin.OptimizeMode.Debug) {
            try self.setupDebugMessenger();
        }

        try self.pickPhysicalDevice();
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
            .enabled_extension_count = @intCast(self.extensions.items.len),
            .pp_enabled_extension_names = self.extensions.items.ptr,
            .enabled_layer_count = 0,
            .flags = .{ .enumerate_portability_bit_khr = true },
            .p_next = null,
        };

        if (builtin.mode == std.builtin.OptimizeMode.Debug) {
            const debug_create_info = createDebugMessengerCreateInfo();
            for (self.extensions.items, 0..) |ext, i| {
                std.log.debug("Required Extension {}: {s}", .{ i, ext });
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

    fn pickPhysicalDevice(self: *Self) !void {
        var device_count: u32 = undefined;
        var result = try self.vki.enumeratePhysicalDevices(self.instance, &device_count, null);
        try VkAssert.withMessage(result, "Failed to find a GPU with Vulkan support.");

        var devices = try self.allocator.alloc(vk.PhysicalDevice, device_count);
        defer self.allocator.free(devices);

        result = try self.vki.enumeratePhysicalDevices(self.instance, &device_count, devices.ptr);
        try VkAssert.withMessage(result, "Failed to find a GPU with Vulkan support.");

        for (devices) |device| {
            if (try isDeviceSuitable(device, &self.vki)) {
                self.physical_device = device;
                break;
            }
        }

        if (self.physical_device == .null_handle) {
            return error.NoSuitableGPU;
        }
    }

    fn getRequiredExtensions(self: *Self) !void {
        const glfw_extensions = glfw.getRequiredInstanceExtensions() orelse return blk: {
            const err = glfw.mustGetError();
            std.log.err("Failed to get required vulkan instance extensions: error={s}", .{err.description});
            break :blk error.code;
        };

        var required_extensions = std.ArrayList([*:0]const u8).init(self.allocator);
        try required_extensions.appendSlice(glfw_extensions);

        if (builtin.os.tag == .macos) {
            try required_extensions.append(vk.extension_info.khr_portability_enumeration.name);
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

        self.extensions = required_extensions;
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

fn isDeviceSuitable(device: vk.PhysicalDevice, dispatch: *InstanceDispatch) !bool {
    _ = dispatch;
    _ = device;
    return true; // At this point we don't care as long as it supports Vulkan at all
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

fn deinitGlfw(window: *glfw.Window) void {
    window.destroy();
    glfw.terminate();
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
