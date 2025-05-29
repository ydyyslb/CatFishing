var phoneinput = document.getElementById("phone");
phoneinput.onblur = checkphone;

//将每一个功能抽象成一个函数
function checkphone() {
    //获取用户输入的手机号
    var phone = phoneinput.value.trim();
    //长度是11,1开头，只能是数字
    var reg = /^[1]\d{10}$/;
    //判断用户输入的手机号是否符合规范（长度为10~12，必须为数字）
    var flag = reg.test(phone);
    if (flag) {
        //符合规则
        document.getElementById("phone_error").style.display = 'none';
    } else {
        //不符合规范
        document.getElementById("phone_error").style.display = '';
    }
    return flag;
}

var password;
//验证密码是否符合规则
//获取密码的输入框对象
var passwordinput = document.getElementById("password");
//获取用户输入的密码
passwordinput.onblur = checkpassword;
function checkpassword() {
    password = passwordinput.value.trim();
    var reg = /^\w{6,12}$/;
    var flag = reg.test(password);
    if (flag) {
        // 符合规则时隐藏错误提示
        document.getElementById("password_error").style.display = 'none';
    } else {
        // 不符合规范时显示错误提示
        document.getElementById("password_error").style.display = '';
    }
    return flag;
}


var register = document.getElementById("register");
register.addEventListener("click", function () {
    //跳转
    window.location.href = "register.html"
})

var regForm = document.getElementById("reg-form2");
var checkbox = document.getElementById("agree");

// 绑定表单的提交事件处理函数
regForm.addEventListener("submit", function (event) {
    // 判断复选框是否被勾选
    if (!checkbox.checked) {
        event.preventDefault(); // 阻止表单的默认提交行为
        alert("Please click Agree to the agreement and register");
    } else {
        event.preventDefault(); // 阻止表单的默认提交行为
        //挨个判断每一个表单项是否符合要求，如果有一个不符合，则返回false
        var flag = checkpassword() && checkphone();
        if (flag) {
            alert("Success");
            window.location.href = "sta_login.html"
        } else {
            alert("Submission failed, please check the input");
            return false;
        }
    }
});


function togglePasswordVisibility() {
    var passwordInput = document.getElementById('password');
    if (passwordInput.type === 'password') {
        passwordInput.type = 'text';
    } else {
        passwordInput.type = 'password';
    }
}

