const builtin = @import("builtin");
const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

const glfw = @import("glfw");
const Hints = glfw.Window.Hints;

const vk = @import("vk.zig");
const vk_ctx = @import("vk_context.zig");
const BaseDispatch = vk_ctx.BaseDispatch;
const InstanceDispatch = vk_ctx.InstanceDispatch;

const width = 800;
const height = 600;

const validation_layers = [_][*:0]const u8{
    "VK_LAYER_KHRONOS_validation",
};

pub const App = struct {
    const Self = @This();

    window: glfw.Window,
    alloc: Allocator,

    instance: vk.Instance,
    extensions: ArrayList([*:0]const u8),

    vkb: BaseDispatch,
    vki: InstanceDispatch,

    pub fn init(alloc: Allocator) !App {
        var window = try initWindow();
        errdefer deinitGlfw(&window);

        const base_dispatch = try BaseDispatch.load(@as(
            vk.PfnGetInstanceProcAddr,
            @ptrCast(&glfw.getInstanceProcAddress),
        ));

        var app = App{
            .alloc = alloc,
            .window = window,
            .vkb = base_dispatch,
            .vki = undefined,
            .extensions = undefined,
            .instance = undefined,
        };

        try app.initVulkan();

        return app;
    }

    pub fn deinit(self: *Self) void {
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
    }

    fn getRequiredExtensions(self: *Self) !void {
        const glfw_extensions = glfw.getRequiredInstanceExtensions() orelse return blk: {
            const err = glfw.mustGetError();
            std.log.err("Failed to get required vulkan instance extensions: error={s}", .{err.description});
            break :blk error.code;
        };

        var required_extensions = std.ArrayList([*:0]const u8).init(self.alloc);
        try required_extensions.appendSlice(glfw_extensions);

        if (builtin.os.tag == .macos) {
            try required_extensions.append(vk.extension_info.khr_portability_enumeration.name);
        }

        if (builtin.mode == std.builtin.OptimizeMode.Debug) {
            var property_count: u32 = 0;
            var result = try self.vkb.enumerateInstanceExtensionProperties(null, &property_count, null);
            if (result != .success) {
                std.log.err("Failed to enumerate instance extension properties", .{});
                return error.EnumerationFailed;
            }

            // Create the buffer to store the supported instance extension properties
            var extension_properties = ArrayList(vk.ExtensionProperties).init(self.alloc);
            try extension_properties.resize(property_count);
            defer extension_properties.deinit();

            result = try self.vkb.enumerateInstanceExtensionProperties(null, &property_count, extension_properties.items.ptr);
            if (result != .success) {
                std.log.err("Failed to enumerate instance extension properties", .{});
                return error.EnumerationFailed;
            }

            for (extension_properties.items) |ext| {
                std.log.debug("Available Extension: {s}", .{ext.extension_name});
            }

            // Ensure that all of the required extensions are supported
            var count = required_extensions.items.len;
            for (extension_properties.items) |found_ext| {
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
        };

        if (builtin.mode == std.builtin.OptimizeMode.Debug) {
            for (self.extensions.items, 0..) |ext, i| {
                std.log.debug("Required Extension {}: {s}", .{ i, ext });
            }

            if (!(try checkValidationLayerSupport(self))) {
                return error.LayerNotPresent;
            }

            create_info.enabled_layer_count = @intCast(validation_layers.len);
            create_info.pp_enabled_layer_names = &validation_layers;
        }

        self.instance = try self.vkb.createInstance(&create_info, null);
        self.vki = try InstanceDispatch.load(self.instance, self.vkb.dispatch.vkGetInstanceProcAddr);
    }

    fn checkValidationLayerSupport(self: *Self) !bool {
        var layer_count: u32 = undefined;
        var result = try self.vkb.enumerateInstanceLayerProperties(&layer_count, null);
        if (result != .success) {
            std.log.err("Failed to enumerate instance layer properties", .{});
            return error.EnumerationFailed;
        }

        var available_layers = std.ArrayList(vk.LayerProperties).init(self.alloc);
        defer available_layers.deinit();

        try available_layers.resize(layer_count);
        result = try self.vkb.enumerateInstanceLayerProperties(&layer_count, available_layers.items.ptr);
        if (result != .success) {
            std.log.err("Failed to enumerate instance layer properties", .{});
            return error.EnumerationFailed;
        }

        for (validation_layers) |layer_name| {
            var layer_found = false;

            for (available_layers.items) |layer_properties| {
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
};

fn deinitGlfw(window: *glfw.Window) void {
    window.destroy();
    glfw.terminate();
}

fn strEql(arr: []const u8, str: [*:0]const u8) bool {
    for (arr, str) |c1, c2| {
        if (c1 != c2) return false;
        if (c1 == 0 or c2 == 0) break;
    }

    return true;
}
