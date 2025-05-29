// let commonURL = "http://192.168.50.115:8081";
let commonURL = "/api";

// const xhr = new XMLHttpRequest();
// const url = '/user/download?name=6885695c-c2ee-4927-9be3-6b53803cdab0.jpg';
// xhr.open('GET', url, true);

// // 在发送请求之前进行拦截处理
// xhr.onreadystatechange = function () {
//     if (xhr.readyState === XMLHttpRequest.OPENED) {
//         // 如果存在token，则设置请求头
//         if (token) {
//             xhr.setRequestHeader('authorization', token);
//         }
//     }
// };

axios.defaults.baseURL = commonURL;
axios.defaults.timeout = 2000;
// request拦截器，将用户token放入头中
let token = sessionStorage.getItem("token");
axios.interceptors.request.use(
    config => {
        if (token) config.headers['authorization'] = token
        return config
    },
    error => {
        console.log(error)
        return Promise.reject(error)
    }
)
axios.interceptors.response.use(function (response) {
    // 判断执行结果
    if (!response.data.success) {
        return Promise.reject(response.data.errorMsg)
    }
    return response.data;
}, function (error) {
    // 一般是服务端异常或者网络异常
    console.log(error)
    if (error.response.status == 401) {
        // 未登录，跳转
        setTimeout(() => {
            location.href = "/Login.html"
        }, 200);
        return Promise.reject("Please login first!");
    }
    return Promise.reject("The server is abnormal");
});

