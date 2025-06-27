use memmap::{Mmap, MmapMut, MmapOptions};

const BLOCK_SIZE: usize = 4096;

#[repr(C)]
struct Block {
    header: Header,
    data: [u8; BLOCK_SIZE],
}

struct Header {
    next: *mut Block,
    prev: *mut Block,
}

struct Allocator {
    map: MmapMut,
}

impl Allocator {
    fn new() -> Self {
        let map = MmapOptions::new().map_anon().unwrap();
        Self { map }
    }

    unsafe fn alloc(&mut self) -> *mut Block {
        let block = self.map.as_mut_ptr() as *mut Block;
        // self.map.advance(BLOCK_SIZE);
        block
    }
}
