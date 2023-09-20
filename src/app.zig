const std = @import("std");
const Allocator = std.mem.Allocator;

const glfw = @import("glfw");
const Hints = glfw.Window.Hints;

const vk = @import("vulkan");
const vk_ctx = @import("vk_context.zig");
const BaseDispatch = vk_ctx.BaseDispatch;

const width = 800;
const height = 600;

pub const App = struct {
    const Self = @This();

    window: glfw.Window,
    alloc: Allocator,

    vkb: BaseDispatch,

    pub fn init(alloc: Allocator) !App {
        const window = try initWindow();
        const base_dispatch = try BaseDispatch.load(@as(
            vk.PfnGetInstanceProcAddr,
            @ptrCast(&glfw.getInstanceProcAddress),
        ));

        var app = App{
            .alloc = alloc,
            .window = window,
            .vkb = base_dispatch,
        };

        return app;
    }

    pub fn deinit(self: *Self) void {
        self.window.destroy();
        glfw.terminate();
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

    fn initVulkan(_: *Self) !void {
        std.log.debug("I'm here!", .{});
    }

    fn createInstance(self: *Self) !void {
        const app_info = vk.ApplicationInfo{
            .p_application_name = "Hello Triangle",
            .application_version = vk.makeApiVersion(1, 0, 0),
            .p_engine_name = "No Engine",
            .engine_version = vk.makeApiVersion(1, 0, 0),
            .api_version = vk.makeApiVersion(1, 0, 0),
        };

        const glfw_extensions = glfw.getRequiredInstanceExtensions() orelse return blk: {
            const err = glfw.mustGetError();
            std.log.err("Failed to get required vulkan instance extensions: error={s}", .{err.description});
            break :blk error.code;
        };

        const create_info = vk.InstanceCreateInfo{
            .p_application_info = &app_info,
            .enabled_extension_count = glfw_extensions.len,
            .pp_enabled_extension_names = glfw_extensions.ptr,
            .enabled_layer_count = 0,
        };

        try self.vkb.createInstance(&create_info, null);
    }
};
