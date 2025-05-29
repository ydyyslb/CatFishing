const DOM = {
  tabsNav: document.querySelector('.tabs__nav'),
  tabsNavItems: document.querySelectorAll('.tabs__nav-item'),
  panels: document.querySelectorAll('.tabs__panel')
};


//set active nav element
const setActiveItem = elem => {

  DOM.tabsNavItems.forEach(el => {

    el.classList.remove('js-active');

  });

  elem.classList.add('js-active');

};

//find active nav element
const findActiveItem = () => {

  let activeIndex = 0;

  for (let i = 0; i < DOM.tabsNavItems.length; i++) {

    if (DOM.tabsNavItems[i].classList.contains('js-active')) {
      activeIndex = i;
      break;
    };

  };

  return activeIndex;

};

//find active nav elements parameters: left coord, width
const findActiveItemParams = activeItemIndex => {

  const activeTab = DOM.tabsNavItems[activeItemIndex];

  //width of elem
  const activeItemWidth = activeTab.offsetWidth - 1;

  //left coord in the tab navigation
  const activeItemOffset_left = activeTab.offsetLeft;

  return [activeItemWidth, activeItemOffset_left];

};

//appending decoration block to an active nav element
const appendDecorationNav = () => {

  //creating decoration element
  let decorationElem = document.createElement('div');

  decorationElem.classList.add('tabs__nav-decoration');
  decorationElem.classList.add('js-decoration');

  //appending decoration element to navigation
  DOM.tabsNav.append(decorationElem);

  //appending styles to decoration element
  return decorationElem;
};

//appending styles to decoration nav element
const styleDecorElem = (elem, decorWidth, top) => {
  elem.style.width = `${decorWidth}px`;
  elem.style.transform = `translateY(${top}px)`;
};

//find active panel
const findActivePanel = index => {
  const panelsLength = DOM.panels.length;
  const newIndex = panelsLength - 1 - index; // 计算新的索引
  return DOM.panels[newIndex];
};

//set active panel class
const setActivePanel = index => {
  DOM.panels.forEach(el => {
    el.classList.remove('js-active');
    el.style.zIndex = '1'; // 将所有 panel 的 z-index 重置为默认值
  });

  const panelToShow = findActivePanel(index);
  panelToShow.classList.add('js-active');
  panelToShow.style.zIndex = '2'; // 设置被激活的 panel 的 z-index 为较高的值
};

//onload function
window.addEventListener('load', () => {

  //find active nav item
  const activeItemIndex = findActiveItem();

  //find active nav item params
  const [decorWidth, decorOffset] = findActiveItemParams(activeItemIndex);

  //appending decoration element to an active elem
  const decorElem = appendDecorationNav();

  //setting styles to the decoration elem
  styleDecorElem(decorElem, decorWidth, decorOffset);

  //find active panel
  findActivePanel(activeItemIndex);

  //set active panel
  setActivePanel(activeItemIndex);
});

//click nav item function
DOM.tabsNav.addEventListener('click', e => {

  const navElemClass = 'tabs__nav-item';

  //check if we click on a nav item
  if (e.target.classList.contains(navElemClass)) {
    const clickedTab = e.target;

    //set active nav item
    setActiveItem(clickedTab);

    //find active nav item
    const activeItemIndex = Array.from(DOM.tabsNavItems).indexOf(clickedTab);

    //find active nav item params
    const [decorWidth, decorOffset] = findActiveItemParams(activeItemIndex);

    //setting styles to the decoration elem
    const decorElem = document.querySelector('.js-decoration');

    // Update decorOffset value here
    const tabsNavRect = DOM.tabsNav.getBoundingClientRect();
    const tabsNavTop = tabsNavRect.top;
    const clickedTabRect = clickedTab.getBoundingClientRect();
    const tabTop = clickedTabRect.top;
    const top = tabTop - tabsNavTop;

    styleDecorElem(decorElem, decorWidth, top);

    //find active panel
    findActivePanel(activeItemIndex);

    //set active panel
    setActivePanel(activeItemIndex);
  }

});

//用户信息修改

// 获取登出按钮元素
var logoutBtn = document.getElementById("logoutBtn");
// 绑定点击事件
logoutBtn.addEventListener("click", function (event) {
  event.preventDefault(); // 阻止默认的链接跳转行为

  // 发起登出请求
  axios.get('/user/users/logout')
    .then(function (response) {
      if (response.data.code === 1) {
        // 登出成功后跳转页面
        window.location.href = "/templates/index.html";
      } else {
        // 返回信息验证失败
        console.log(response.data.msg);
      }
    })
    .catch(function (error) {
      console.log(error);
    });
});

