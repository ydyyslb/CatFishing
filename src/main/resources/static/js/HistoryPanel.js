new Vue({
    el: "#panel6",
    data() {
        return {
            multipleSelection: [],
            counts_history: 0,
            page_history: 1,
            pageSize_history: 10,
            loading: true,
            user: {},
            text: [],
            tests:[]
        }
    },
    // beforeMount() {
    //     this.fetchUserInfo();

    // },
    mounted() {
        this.init();
    },
    computed: {
        formatteddate() {
            return function (row) {
                const date = new Date(row.recordTime);
                const year = date.getFullYear();
                const month = (date.getMonth() + 1).toString();
                const day = date.getDate().toString();
                const hours = date.getHours().toString().padStart(2, '0');
                const minutes = date.getMinutes().toString().padStart(2, '0');
                const seconds = date.getSeconds().toString().padStart(2, '0');
                return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
            };
        },

    },
    methods: {
        // filterHandler(value, row, column) {
        //     // const formattedValue = this.formatTask(row, column);
        //     return formattedValue === value;
        // },
        async GetQuestions(ids) {
            try {
                const response = await axios.get('/question/getQuestionByID', {
                    params: {
                        questionids:ids
                    },
                });

                if (response.success) {
                    
                    return response.data;
                }

            } catch (error) {
                // 处理错误情况
                console.error('Error fetching image:', error);
            }
        },
        async init() {
            try {
                this.tests = [];
                const data = {
                    page: this.page_history,
                    pageSize: this.pageSize_history
                };

                const response = await axios.post('/test/page', data);

                if (response.success) {
                    this.text = response.data.records || [];
                    this.counts_history = response.data.total;
                   
                   
                    for (let index = 0; index < this.text.length; index++) {
                        const item = this.text[index];                      
                        const task_questions = await this.GetQuestions(item.questionId); // 等待异步调用完成
                        const formattedQuestions = task_questions.map((question, index) => {
                            return `${index + 1}. ${question.question}`;
                        });
                      
                        
                        const formattedRightAnswer = task_questions.map((question, index) => {
                            let choicesString = question.choices.replace(/'/g, '').replace('[', '').replace(']', '');
                            let choicesArray = choicesString.split(',').map(choice => choice.trim());
                            return `${index + 1}. ${choicesArray[question.answer]}`
                        });

                        const formattedUserAnswer = task_questions.map((question, index) => {
                            let choicesString = question.choices.replace(/'/g, '').replace('[', '').replace(']', '');
                            let choicesArray = choicesString.split(',').map(choice => choice.trim());
                            return `${index + 1}. ${choicesArray[item.userAnswer[index]]}`
                        });
                        let test = {
                            recordTime: item.finishTime,
                            task: item.task,
                            questions: formattedQuestions.join(' '),
                            right_answer: formattedRightAnswer.join(' '),
                            user_answer: formattedUserAnswer.join(' '),
                            score: item.userScore,
                            correct_number: item.correctNumber,
                            wrong_number: item.wrongNumber,
                            id: item.id
                        };

                        this.tests.push(test);
                    }
                }
            } catch (error) {
                this.$message.error('Something Wrong：' + error);
            }
        },
        handleSizeChange(val) {
            this.pageSize_history = val
            this.init()
        },
        handleCurrentChange(val) {
            this.page_history = val
            this.init()
        },
        toggleSelection(rows) {
            if (rows) {
                rows.forEach(row => {
                    this.$refs.multipleTable.toggleRowSelection(row);
                });
            } else {
                this.$refs.multipleTable.clearSelection();
            }
        },
        handleSelectionChange(val) {
            this.multipleSelection = val;
        },
        async handleDeleteSelection() {
            const selectedRows = this.multipleSelection; // 获取选中的行数据
            // 遍历选中的行数据，逐个发送删除请求
            try {
                for (const row of selectedRows) {
                    const res = await axios.post('/test/delete', row);
                    
                }
                this.init(); // 删除完成后进行页面数据初始化操作
                this.$message.success('Delete successful!');
            } catch (error) {
                console.error(error);
                this.$message.error('Delete failed: ' + error.message);
            }
        },
        async handleDelete(row) {
            try {
                const res = await axios.post('/test/delete', row); // 发送包含整个对象的 POST 请求
                
                
                this.$message.success('Delete successful!');
                this.init();
            } catch (error) {
                console.error(error);
            }
        },
        formatter(row, column) {
            return row[column.property];
        },
        formatText(row, column) {
            const text = row[column.property];
            if (text.length > 50) {
                return text.substring(0, 50) + "...";  // 如果文本长度超过10个字符，只显示前10个字符并加上省略号
            } else {
                return text;
            }
        },
    }
})
