/**
 * WEBSITE: https://themefisher.com
 * TWITTER: https://twitter.com/themefisher
 * FACEBOOK: https://www.facebook.com/themefisher
 * GITHUB: https://github.com/themefisher/
 */

/* ====== Index ======
1. SPLINA AREA CHART 01
2. SPLINA AREA CHART 02
3. SPLINA AREA CHART 03
4. SPLINA AREA CHART 04
5. MIXED CHART 01
6. RADIAL BAR CHART 01
7.1 HORIZONTAL BAR CHART
7.2 HORIZONTAL BAR CHART2
8.1 TABLE SMALL BAR CHART 01
8.2 TABLE SMALL BAR CHART 02
8.3 TABLE SMALL BAR CHART 03
8.4 TABLE SMALL BAR CHART 04
8.5 TABLE SMALL BAR CHART 05
8.6 TABLE SMALL BAR CHART 06
8.7 TABLE SMALL BAR CHART 07
8.8 TABLE SMALL BAR CHART 08
8.9 TABLE SMALL BAR CHART 09
8.10 TABLE SMALL BAR CHART 10
8.11 TABLE SMALL BAR CHART 11
8.12 TABLE SMALL BAR CHART 12
8.13 TABLE SMALL BAR CHART 13
8.14 TABLE SMALL BAR CHART 14
8.15 TABLE SMALL BAR CHART 15
9.1 STATUS SMALL BAR CHART 01
9.2 STATUS SMALL BAR CHART 02
9.3 STATUS SMALL BAR CHART 03
10.1 LINE CHART 01
10.2 LINE CHART 02
10.3 LINE CHART 03
10.4 LINE CHART 04
11.1 BAR CHART LARGE 01
11.2 BAR CHART LARGE 02
12.1 DONUT CHART 01
12.2 DONUT CHART 02
13. PIE CHART
14. RADER CHART
15.1 ARIA CHART 01

====== End ======*/

"use strict";

/*======== 10.1 LINE CHART 01 ========*/
// 假设的正确率数据序列
var correctnessDataSeries = [50, 80, 94, 60, 85, 95, 97];

// 获取过去七天的日期，并格式化为 "M/d" 的格式
function getLastSevenDaysFormatted() {
  var dates = [];
  for (var i = 6; i >= 0; i--) {
    var d = new Date();
    d.setDate(d.getDate() - i);
    dates.push((d.getMonth() + 1) + '/' + d.getDate());
  }
  return dates;
}
var datesFormatted = getLastSevenDaysFormatted(); // 获取格式化的过去七天的日期数组

var lineChartOption1 = {
  chart: {
    height: 350,
    type: "line",
    toolbar: {
      show: false,
    },
    
  },
  stroke: {
    width: 3,
    curve: "smooth",
  },
  colors: ["#4CAF50"], // 绿色表示正确率
  series: [
    {
      name: "natural science",
      data: correctnessDataSeries // 使用正确率数据序列
    }
  ],
  labels: datesFormatted, // 使用格式化的过去七天的日期
  markers: {
    size: 5,
  },
  xaxis: {
    // X轴标签显示
    type: 'category',
    categories: datesFormatted, // 将横坐标日期格式化
  },
  yaxis: {
    // Y轴设置固定梯度
    tickAmount: 5, // 因为包括 0，所以这里是 5
    min: 0,
    max: 100,
    labels: {
      formatter: function (val) {
        return val.toFixed(0); // 没有小数位
      }
    }
  },
  tooltip: {
    theme: "dark",
    x: {
      show: true,
      format: 'M/d'
    },
    y: {
      formatter: function (val) {
        return val.toFixed(1) + "%"; // 提示框中的百分比值带有一位小数
      },
    },
  },
  legend: {
    show: true,
  },
};

// 获取DOM元素
var lineChart1 = document.querySelector("#line-chart-1");
// 如果元素存在，创建图表实例
if (lineChart1 !== null) {
  var randerLineChart1 = new ApexCharts(lineChart1, lineChartOption1);
  randerLineChart1.render();
}

/*======== 10.2 LINE CHART 02 ========*/
// 假设的正确率数据序列
var correctnessDataSeries = [50, 80, 94, 60, 85, 95, 97];

// 获取过去七天的日期，并格式化为 "M/d" 的格式
function getLastSevenDaysFormatted() {
  var dates = [];
  for (var i = 6; i >= 0; i--) {
    var d = new Date();
    d.setDate(d.getDate() - i);
    dates.push((d.getMonth() + 1) + '/' + d.getDate());
  }
  return dates;
}
var datesFormatted = getLastSevenDaysFormatted(); // 获取格式化的过去七天的日期数组

var lineChartOption2 = {
  chart: {
    height: 350,
    type: "line",
    toolbar: {
      show: false,
    },
    
  },
  stroke: {
    width: 3,
    curve: "smooth",
  },
  colors: ["#9e6de0"], // 绿色表示正确率
  series: [
    {
      name: "language science",
      data: correctnessDataSeries // 使用正确率数据序列
    }
  ],
  labels: datesFormatted, // 使用格式化的过去七天的日期
  markers: {
    size: 5,
  },
  xaxis: {
    // X轴标签显示
    type: 'category',
    categories: datesFormatted, // 将横坐标日期格式化
  },
  yaxis: {
    // Y轴设置固定梯度
    tickAmount: 5, // 因为包括 0，所以这里是 5
    min: 0,
    max: 100,
    labels: {
      formatter: function (val) {
        return val.toFixed(0); // 没有小数位
      }
    }
  },
  tooltip: {
    theme: "dark",
    x: {
      show: true,
      format: 'M/d'
    },
    y: {
      formatter: function (val) {
        return val.toFixed(1) + "%"; // 提示框中的百分比值带有一位小数
      },
    },
  },
  legend: {
    show: true,
  },
};

// 获取DOM元素
var lineChart2 = document.querySelector("#line-chart-2");
// 如果元素存在，创建图表实例
if (lineChart1 !== null) {
  var randerLineChart2 = new ApexCharts(lineChart2, lineChartOption2);
  randerLineChart2.render();
}

/*======== 10.3 LINE CHART 03 ========*/
var lineChart = document.querySelector("#line-chart-3");
if (lineChart !== null) {
  // 计算过去七天的日期
  var pastSevenDays = [];
  for (let i = 6; i >= 0; i--) {
    let d = new Date();
    d.setDate(d.getDate() - i);
    pastSevenDays.push(`${d.getMonth() + 1}/${d.getDate()}`);
  }

  var lineChartOptions = {
    chart: {
      height: 350,
      type: 'line', // 更改为折线图
      toolbar: {
        show: false,
      },
    },
    stroke: {
      width:3,
      curve: 'smooth'
    },
    markers: {
      size: 5,
    },
    series: [
      {
        name: "social science",
        data: [3, 9, 12, 24, 14, 11, 26], // 保留一组数据
      }
    ],
    xaxis: {
      categories: pastSevenDays, // 更新横坐标为过去七天的日期
    },
    yaxis: {
      min: 0, // 设置纵坐标最小值
      max: 100, // 设置纵坐标最大值
    },
    tooltip: {
      theme: 'dark',
      shared: true,
      intersect: false,
      x: { show: false }
    },
    legend: {
      show: false,
    },
  };

  var lineChartRender = new ApexCharts(lineChart, lineChartOptions);
  lineChartRender.render();
}

/*======== 10.4 LINE CHART 04 ========*/
var lineChart4 = document.querySelector("#line-chart-4");
if (lineChart4 !== null) {
  var lineChartOption4 = {
    chart: {
      height: 350,
      type: "line",
      toolbar: {
        show: false,
      },
    },
    stroke: {
      width: [2, 3],
      curve: "smooth",
      dashArray: [0, 5],
    },
    plotOptions: {
      horizontal: false,
    },

    colors: ["#9e6de0", "#fec400"],

    legend: {
      show: true,
      position: "top",
      horizontalAlign: "right",
      markers: {
        width: 20,
        height: 5,
        radius: 0,
      },
    },
    series: [
      {
        data: [6, 10, 8, 20, 15, 6, 21],
      },
      {
        data: [8, 6, 15, 10, 25, 8, 32],
      },
    ],
    labels: [
      "04 jan",
      "05 jan",
      "06 jan",
      "07 jan",
      "08 jan",
      "09 jan",
      "10 jan",
    ],
    markers: {
      size: [5, 0],
    },
    xaxis: {
      axisBorder: {
        show: false,
      },
      axisTicks: {
        show: false,
      },
    },
    tooltip: {
      theme: "dark",
      shared: true,
      intersect: false,
      fixed: {
        enabled: false,
      },
      x: {
        show: false,
      },
      y: {
        title: {
          formatter: (labels) => labels,
        },
      },
      marker: {
        show: true,
      },
    },
  };
  var randerLineChart4 = new ApexCharts(lineChart4, lineChartOption4);
  randerLineChart4.render();
}

/*======== 11.1 BAR CHART LARGE 01 ========*/
var barChartLg1 = document.querySelector("#barchartlg1");
if (barChartLg1 !== null) {
  var barChartOptions1 = {
    chart: {
      height: 275,
      type: "bar",
      toolbar: {
        show: false,
      },
    },
    colors: ["#faafca"],
    plotOptions: {
      bar: {
        horizontal: false,
        endingShape: "flat",
        columnWidth: "55%",
      },
    },
    legend: {
      position: "bottom",
      horizontalAlign: "left",
    },
    dataLabels: {
      enabled: false,
    },
    stroke: {
      show: true,
      width: 2,
      colors: ["transparent"],
    },
    series: [
      {
        name: "Referral",
        data: [76, 85, 79, 88, 87, 65],
      },
    ],
    xaxis: {
      categories: ["4 Jan", "5 Jan", "6 Jan", "7 Jan", "8 Jan", "9 Jan"],
    },
    yaxis: {
      show: false,
    },
    fill: {
      opacity: 1,
    },
    tooltip: {
      theme: "dark",
      x: {
        show: false,
      },
      y: {
        formatter: function (val) {
          return  val+ "min";
        },
      },
      marker: {
        show: true,
      },
    },
  };
  var randerBarChartLg1 = new ApexCharts(barChartLg1, barChartOptions1);
  randerBarChartLg1.render();

  var items = document.querySelectorAll(
    "#user-acquisition .nav-underline-active-primary .nav-item"
  );
  items.forEach(function (item, index) {
    item.addEventListener('click', function() {
      if (index === 0) {
        randerBarChartLg1.updateOptions({
          colors: ['#faafca']
        });
        randerBarChartLg1.updateSeries([
          {
            name: "natural science",
            data: [76, 85, 79, 88, 87, 65]
          }
        ]);
      } else if (index === 1) {
        randerBarChartLg1.updateOptions({
          colors: ['#00E396']
        });
        randerBarChartLg1.updateSeries([
          {
            name: "language science",
            data: [66, 50, 35, 52, 52, 45]
          }
        ]);
      } else if (index === 2) {
        randerBarChartLg1.updateOptions({
          colors: ['#008FFB']
        });
        randerBarChartLg1.updateSeries([
          {
            name: "social science",
            data: [64, 64, 58, 45, 77, 53]
          }
        ]);
      }
    });
  });
}

/*======== 11.2 BAR CHART LARGE 02 ========*/
var barChartLg2 = document.querySelector("#barchartlg2");
if (barChartLg2 !== null) {
  var trigoStrength = 3;
  var iteration = 11;

  function getRandom() {
    var i = iteration;
    return (
      (Math.sin(i / trigoStrength) * (i / trigoStrength) +
        i / trigoStrength +
        1) *
      (trigoStrength * 2)
    );
  }

  function getRangeRandom(yrange) {
    return (
      Math.floor(Math.random() * (yrange.max - yrange.min + 1)) + yrange.min
    );
  }

  function generateMinuteWiseTimeSeries(baseval, count, yrange) {
    var i = 0;
    var series = [];
    while (i < count) {
      var x = baseval;
      var y =
        (Math.sin(i / trigoStrength) * (i / trigoStrength) +
          i / trigoStrength +
          1) *
        (trigoStrength * 2);

      series.push([x, y]);
      baseval += 300000;
      i++;
    }
    return series;
  }

  var optionsColumn = {
    chart: {
      height: 315,
      type: "bar",
      toolbar: {
        show: false,
      },
      animations: {
        enabled: true,
        easing: "linear",
        dynamicAnimation: {
          speed: 1000,
        },
      },

      events: {
        animationEnd: function (chartCtx) {
          const newData = chartCtx.w.config.series[0].data.slice();
          newData.shift();
          window.setTimeout(function () {
            chartCtx.updateOptions(
              {
                series: [
                  {
                    name: "Load Average",
                    data: newData,
                  },
                ],
                xaxis: {
                  min: chartCtx.minX,
                  max: chartCtx.maxX,
                },
                subtitle: {
                  text: parseInt(
                    getRangeRandom({ min: 1, max: 20 })
                  ).toString(),
                },
              },
              false,
              false
            );
          }, 300);
        },
      },
      toolbar: {
        show: false,
      },
      zoom: {
        enabled: false,
      },
    },
    dataLabels: {
      enabled: false,
    },
    stroke: {
      width: 0,
    },
    colors: "#9e6de0",
    series: [
      {
        name: "Load Average",
        data: generateMinuteWiseTimeSeries(
          new Date("12/12/2016 00:20:00").getTime(),
          12,
          {
            min: 10,
            max: 110,
          }
        ),
      },
    ],
    title: {
      text: "Ave Page views per minute",
      align: "left",
      offsetY: 35,
      style: {
        fontSize: "12px",
        color: "#8a909d",
      },
    },
    subtitle: {
      text: "20%",
      floating: false,
      align: "left",
      offsetY: 0,
      style: {
        fontSize: "22px",
        color: "#9e6de0",
      },
    },
    fill: {
      type: "solid",
      colors: "#9e6de0",
    },
    xaxis: {
      type: "datetime",
      range: 2700000,
    },
    legend: {
      show: false,
    },
    tooltip: {
      theme: "dark",
      x: {
        show: false,
      },
      y: {
        formatter: function (val) {
          return val;
        },
      },
      marker: {
        show: true,
      },
    },
  };

  var chartColumn = new ApexCharts(barChartLg2, optionsColumn);
  chartColumn.render();

  window.setInterval(function () {
    iteration++;

    chartColumn.updateSeries([
      {
        name: "Load Average",
        data: [
          ...chartColumn.w.config.series[0].data,
          [chartColumn.w.globals.maxX + 210000, getRandom()],
        ],
      },
    ]);
  }, 5000);
}

/*======== 12.1 DONUT CHART 01 ========*/
var donutChart1 = document.querySelector("#donut-chart-1");
if (donutChart1 !== null) {
  var donutChartOptions1 = {
    chart: {
      type: "donut",
      height: 296,
    },

    colors: ["#bb91f2", "#af81eb", "#9e6de0"],
    labels: ["natural science", "language science", "social science"],
    series: [45, 30, 25],
    legend: {
      show: false,
    },
    dataLabels: {
      enabled: false,
    },
    tooltip: {
      y: {
        formatter: function (val) {
          return +val + "%";
        },
      },
    },
  };

  var randerDonutchart1 = new ApexCharts(donutChart1, donutChartOptions1);

  randerDonutchart1.render();
}
/*======== 12.2 Bar chart 03 ========*/
// 创建一个图表实例
var donutChart1 = document.querySelector("#barchartlg3");

// 计算过去十天的日期
var pastTenDays = [];
for (let i = 9; i >= 0; i--) {
  let d = new Date();
  d.setDate(d.getDate() - i);
  pastTenDays.push(`${d.getMonth() + 1}/${d.getDate()}`);
}

var options = {
  chart: {
    type: 'bar',
    height: 315
  },
  series: [{
    name: 'Number',
    data: [30, 40, 35, 50, 49, 60, 70, 91, 125] // 这里需要确保数据的长度为10，对应过去十天的日期
  }],
  xaxis: {
    categories: pastTenDays // 更新为过去十天的日期
  },
  yaxis: {
    min: 0, // 纵坐标的最小值
    max: 250, // 纵坐标的最大值
    tickAmount: 5 // 包括最小值和最大值的刻度数量，间隔固定，产生0, 50, 100, 150, 200, 250的值
  }
}

var chart = new ApexCharts(document.querySelector("#barchartlg3"), options);
chart.render();
/*======== 10.5 LINE CHART 05 ========*/
// 创建一个图表实例
var lineChart = document.querySelector("#line-chart-5");

// 计算过去十天的日期
var pastTenDays = [];
for (let i = 9; i >= 0; i--) {
  let d = new Date();
  d.setDate(d.getDate() - i);
  pastTenDays.push(`${d.getMonth() + 1}/${d.getDate()}`);
}

var options = {
  chart: {
    type: 'line', // 更改为折线图
    height: 315
  },
  stroke: {
    curve: 'smooth',
    dashArray: [0, 5] // 这将给第二个系列（虚线）设置一个虚线的样式
  },
  series: [
    {
      name: 'Review Count', // 实线系列
      data: [10, 20, 10, 5, 20, 3, 5, 5, 3, 4] // 确保包含10个数据点
    },
    {
      name: 'Number of Incorrect Questions', // 虚线系列
      data: [5, 15, 5, 20, 15, 5, 20, 3, 5, 5] // 确保包含10个数据点
    }
  ],
  xaxis: {
    categories: pastTenDays // 使用过去十天的日期
  },
  yaxis: {
    min: 0, // Y轴的最小值
    max: 20, // Y轴的最大值
    tickAmount: 4 // 包括最小值和最大值的刻度数量，这会产生0, 5, 10, 15, 20的值
  },
  legend: {
    position: 'bottom', // 图例位置
    offsetY: -10 // 图例与X轴的垂直偏移量
  }
}

var chart = new ApexCharts(lineChart, options);
chart.render();