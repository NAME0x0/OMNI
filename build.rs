fn main() {
    if std::env::var_os("CARGO_FEATURE_NATIVE_MANIFOLD").is_some() {
        cc::Build::new()
            .file("src/kernels/native/manifold_fold.c")
            .compile("perspective_manifold_fold");

        cc::Build::new()
            .cpp(true)
            .file("src/kernels/native/manifold_nearest.cpp")
            .compile("perspective_manifold_nearest");
    }
}
