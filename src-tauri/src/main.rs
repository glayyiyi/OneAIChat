#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use oneai_chat_lib;

fn main() {
  oneai_chat_lib::run();
}
