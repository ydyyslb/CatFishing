
new Vue({
    el: "#userInfo_Panel",
    data() {
        return {
            editing: false,//判断是否为编辑状态
            loading: true,
            user: {},
            Occupation_value: '',
            
            Occupation_options: [{
                value: 'student',
                label: 'student'
            }, {
                value: 'teacher',
                label: 'teacher'
            }, {
                value: 'other',
                label: 'other'
                }],
            Occupation_value:'',
            Gender_options: [{
                value: 'Female',
                label: 'Female'
            }, {
                value: 'Male',
                label: 'Male'
            }, {
                value: 'Other',
                label: 'Other'
            }],
            Gender_value: '',
            Country_options:[{
                value: 'China',
                label: 'China'
            }, {
                value: 'America',
                label: 'America'
            }, {
                value: 'India',
                label: 'India'
            }, {
                value: 'Italian',
                label: 'Italian'
            }, {
                value: 'Britain',
                label: 'Britain'
            }],
            Country_value: ''
        }
    },
    beforeMount() {
        this.fetchUserInfo();
    },
    computed: {
        formattedBirthdate() {
            const birthdate = new Date(this.user.birthdate);
            const year = birthdate.getFullYear();
            const month = (birthdate.getMonth() + 1).toString().padStart(2, '0');
            const day = birthdate.getDate().toString().padStart(2, '0');
            return `${year}-${month}-${day}`;
        },
        calculateAge() {
            const birthYear = new Date(this.user.birthdate).getFullYear();
            const currentYear = new Date().getFullYear();
            return currentYear - birthYear;
        }
    },
    watch: {
        'user.birthdate': function (newDate) {
            this.user.age = this.calculateAge;
        }
    },
    methods: {
        async fetchUserInfo() {
            const _this = this;
            try {
                const response = await axios.get('/user/me');
                if (response.success) {
                    _this.user = response.data;
                    _this.loading = false;
                }
                else {
                    _this.loading = false;
                    window.location.href = './Login.html'
                    throw new Error('Network response was not ok.');
                }
            } catch (error) {
                this.$message.error(error);
            } 
        },

        cancellation() {
            axios.post('/user/users/cancel', this.user).then(response => {
                if (response.data.code === 1) {
                    this.$message.success('cancel successfully');
                    window.location.href = './index.html'
                } else {
                    this.$message.error('Failed to cancel');
                }
            }).catch(error => {
                console.error(error);
            });
        },

        saveUser() {
            axios.put('/user/update', this.user).then(response => {
                if (response.success) {
                    this.editing = false;
                    this.$message.success('Saved successfully');
                } else {
                    this.$message.error('Failed to save');
                }
            }).catch(error => {
                console.error(error);
            });
        },

        CancelEdit() {
            this.editing = false;
        },
        handleAvatarSuccess(res, file) {
            this.user.imageUrl = `/user/download?name=${encodeURIComponent(res.data)}`;
        },
        beforeAvatarUpload(file) {
            const isJPG = file.type === 'image/jpeg';
            const isLt2M = file.size / 1024 / 1024 < 2;
            if (!isJPG) {
                this.$message.error('Upload avatar images in JPG format only!');
            }
            if (!isLt2M) {
                this.$message.error('The size of the uploaded avatar image cannot exceed 2MB!');
            }
            return isJPG && isLt2M;
        },
    }
})
