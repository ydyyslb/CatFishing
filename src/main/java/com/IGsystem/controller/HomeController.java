package com.IGsystem.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;

@Controller
public class HomeController {

    @RequestMapping("/")
    public String home() {
        // 直接转发到 static 下的 index.html
        return "forward:/index.html";
    }
}
