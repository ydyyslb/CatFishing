
// 发送axios请求函数
function fetchProblems(task = 'All', grade = 'All', subject = 'All', topic = 'All', category = 'All') {
    sessionStorage.setItem('task', task);
    sessionStorage.setItem('grade', grade);
    sessionStorage.setItem('subject', subject);
    sessionStorage.setItem('topic', topic);
    sessionStorage.setItem('category', category);
    // const url = `http://localhost:8080/problems?task=${task}&grade=${grade}&subject=${subject}&topic=${topic}&category=${category}`;

    // axios.get(url)
    //     .then(response => {
    //         console.log(response.data);
    //     })
    //     .catch(error => {
    //         console.error('There was an error!', error);
    //     })
    //     .then(() => {
    //         console.log("已经发送了一次请求")
    //     });
}
