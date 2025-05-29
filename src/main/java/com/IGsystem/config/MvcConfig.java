package com.IGsystem.config;

import com.IGsystem.utils.LoginIntecepter;
import com.IGsystem.utils.RefreshTokenInteceptor;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.web.servlet.config.annotation.InterceptorRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

import javax.annotation.Resource;

@Configuration
public class MvcConfig implements WebMvcConfigurer {

    @Resource
    private StringRedisTemplate stringRedisTemplate;

    @Override
    public void addInterceptors(InterceptorRegistry registry) {
        // token刷新拦截器：放在最前面，拦所有请求，优先刷新token
        registry.addInterceptor(new RefreshTokenInteceptor(stringRedisTemplate))
                .addPathPatterns("/**")
                .order(0);

        // 登录拦截器：只拦需要登录才能访问的路径
        registry.addInterceptor(new LoginIntecepter())
                .addPathPatterns("/**") // 默认拦全部
                .excludePathPatterns(
                        "/index.html",      // 首页HTML
                        "/**/*.html",       // 所有HTML页面
                        "/css/**",          // 静态资源
                        "/js/**",
                        "/image/**",
                        "/fonts/**",
                        "/node_modules/**",
                        "/plugins/**",
                        "/api/user/login",   // 登录接口
                        "/api/user/register",// 注册接口
                        "/api/user/validateName", // 校验用户名
                        "/api/user/download", // 下载接口
                        "/api/question/getImage" // 获取图片接口
                )
                .order(1);
    }
}
