// charts.js

// Function to create a pie chart
function createPieChart(elementId, data) {
    var ctx = document.getElementById(elementId).getContext('2d');
    return new Chart(ctx, {
        type: 'pie',
        data: data,
    });
}
