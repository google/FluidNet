package = "tfluids"
version = "1-00"

source = {
   url = "",
}

description = {
   summary = "Torch fluids utility library",
   detailed = [[
   ]],
   homepage = "",
   license = "Apache V2.0"
}

dependencies = {
   "torch >= 7.0",
   "luaffi"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DWITH_OPENGL=ON -DWITH_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" -DLUA_INCDIR="$(LUA_INCDIR)" -DLUA_LIBDIR="$(LUA_LIBDIR)" && $(MAKE)
]],
   install_command = "cd build && $(MAKE) install"
}
