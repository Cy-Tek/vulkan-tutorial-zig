# Vulkan Tutorial Zig

This is a port to Zig of the [excellent tutorial book by the Khronos Group](https://vulkan-tutorial.com) for getting started with Vulkan. I am new overall to the Zig programming language, so please feel free to let me know anything I can improve on in terms of my coding style!

## Getting Started

To get this project running on your machine, you should just need to install the [Vulkan SDK](https://www.lunarg.com/vulkan-sdk/) and the latest zig master branch (although I think this code should also be compatible with 0.11).
From there, just run `zig build run` to see a window pop up on your machine. This is very much a work in progress, I've ported the first 139 pages as of Sep 26, 2023, so be on the lookout for frequent changes in the future.

You should now see a triangle rendering on your screen if you've pulled the master branch.

I'll do my best to update this README as more progress gets made.

Additionally, if you are trying to follow along with the book, I will have each major chapter under a specific branch for you to look at. For example, if you are looking for the chapter *Drawing a Triangle > Setup*, then the branch name would be `drawing-a-triangle/setup`.

I hope this helps, and have fun with Zig and Vulkan!
