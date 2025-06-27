trait Allocator {
    unsafe fn alloc(&mut self, size: usize, align: usize) -> *mut u8;
    unsafe fn free(&mut self, ptr: *mut u8);
}
