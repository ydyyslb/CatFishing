

new Vue({
    el: "#panel2",
    data() {
        return {
            dialogFormVisible: false,
            form: {
                name: '',
                describe:'',
            },
            formLabelWidth: '120px',
            centerDialogVisible:false,
            editableTabsValue: '2',
            editableTabs: [],
            tabIndex: 1,
            currentPage4: 4,
            tableData: [],
            questionsMap: {}
        }
    },
    computed: {
        isButtonDisabled() {
            return !(this.form.name.trim());
        },
        isButtonPlain() {
            return !(this.form.name.trim());
        }
    },
    mounted() {
        this.init();
    },
    methods: {
        init() {
            axios.get('/user/getfolder')
                .then(response => {

                    const foldersData = response.data;
                    console.log(foldersData.data)
                    this.editableTabs = foldersData.data.map(data => ({
                        title: data.folder.name,
                        name: data.folder.id.toString(),
                        content: data.folder.description
                    }));
                    
                    this.tabIndex = foldersData.defaultTabIndex.toString();
                    this.tableData = foldersData.data.map(data => {
                        
                        return {
                            folderId: data.folder.id,
                            questions: data.questions.map(question => ({
                                date: question.subject,
                                name: question.task,
                                address: question.question
                            }))
                        };

                    });
                    this.questionsMap = {};
                    this.tableData.forEach(item => {
                        this.questionsMap[item.folderId] = item.questions;
                    });
                    
                })
                
                .catch(error => {
                    
                    console.error('Error fetching folder data:', error);
                });
        },
        openDialog() {
            this.centerDialogVisible = !this.centerDialogVisible;           
            document.addEventListener('click', this.handleClickOutside);
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
        closeDialog() {
            this.centerDialogVisible = !this.centerDialogVisible;
            console.log(this.centerDialogVisible)
            document.removeEventListener('click', this.handleClickOutside);
        },
        async confirmAction() {
            //处理确定按钮点击事件的逻辑
            const folderName = this.form.name.trim();
            const description = this.form.describe.trim();
            this.centerDialogVisible = !this.centerDialogVisible;
            this.addTab(folderName);
            try {
                const response = await axios.get('/user/addFolder', {
                    params: {
                        folderName: folderName,
                        description: description
                    },
                });
                if (response.success) {
                    console.log("success");
                }
            } catch (error) {

                this.$message.error(error);
                console.error(error);
            }
        },
        addTab(targetName) {
            let newTabName = ++this.tabIndex + '';
            this.editableTabs.push({
                title: targetName,
                name: newTabName,
                content: '快去收藏些题目吧~'
            });
            this.editableTabsValue = newTabName;
            
        },
        removeTab(targetName) {
            console.log(targetName)
            let tabs = this.editableTabs;
            let activeName = this.editableTabsValue;
            if (activeName === targetName) {
                tabs.forEach((tab, index) => {
                    if (tab.name === targetName) {
                        let nextTab = tabs[index + 1] || tabs[index - 1];
                        if (nextTab) {
                            activeName = nextTab.name;
                        }
                    }
                });
            }

            this.editableTabsValue = activeName;
            this.editableTabs = tabs.filter(tab => tab.name !== targetName);

            // 发送DELETE请求到后端
            axios.delete('/user/removeFolder', { data: { folderId: targetName } })
                .then(response => {
                    // 根据后端返回结果进行相应处理
                    console.log(response.data);
                })
                .catch(error => {
                    console.error('删除收藏夹失败:', error);
                });
        },
        handleSizeChange(val) {
            console.log(`每页 ${val} 条`);
        },
        handleCurrentChange(val) {
            console.log(`当前页: ${val}`);
        }
    }

})
