#[repr(C)]
pub struct String {
    ptr: *const u8,
    len: usize,
}

#[no_mangle]
pub extern "C" fn make_string(ptr: *const u8, len: usize) -> String {
    String { ptr: ptr, len: len }
    // Box::into_raw(Box::new(String { ptr: ptr, len: len }))
}

pub fn as_str<'a>(data: *const u8, length: usize) -> &'a str {
    unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(data, length)) }
}

#[no_mangle]
pub extern "C" fn print_string(s: String) {
    // let s = unsafe { Box::from_raw(s) };
    let str = as_str(s.ptr, s.len);
    println!("{}", str);
}
