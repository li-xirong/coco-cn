$(document).ready(function(){
    $(".form_datetime").datetimepicker({
        format: "yyyy mm dd - hh:ii",
        autoclose: true,
        todayBtn: true,
        startDate: "2017-02-14 10:00",
        minuteStep: 10
    });
})

function user(view){
    user = $(view).attr("value");
    window.location.href = "/history?user=" + user;
}

function searchByTime(){
    var start = $("#start-time");
    var end = $("#end-time");

    var start_mark = 0;
    var start_year = 0;
    var start_month = 0;
    var start_date = 0;
    var start_hour = 0;
    var start_minute = 0;
    if(start.val() != ''){
        var string = start.val();
        start_mark = 1;
        start_year = parseInt(string.substring(0, 4));
        start_month = parseInt(string.substring(5, 7));
        start_date = parseInt(string.substring(8, 10));
        start_hour = parseInt(string.substring(13, 15));
        start_minute = parseInt(string.substring(17, 19));
    }
    var end_mark = 0;
    var end_year = 0;
    var end_month = 0;
    var end_date = 0;
    var end_hour = 0;
    var end_minute = 0;
    if(end.val() != ''){
        var string = end.val();
        end_mark = 1;
        end_year = parseInt(string.substring(0, 4));
        end_month = parseInt(string.substring(5, 7));
        end_date = parseInt(string.substring(8, 10));
        end_hour = parseInt(string.substring(13, 15));
        end_minute = parseInt(string.substring(17, 19));
    }

    $.ajax({
        url:"",
	    data:{
	        start_mark : start_mark,
	    	start_year : start_year,
	    	start_month : start_month,
	    	start_date : start_date,
	    	start_hour : start_hour,
	    	start_minute : start_minute,
	    	end_mark : end_mark,
	    	end_year : end_year,
	    	end_month : end_month,
	    	end_date : end_date,
	    	end_hour : end_hour,
	    	end_minute : end_minute,
	    },
	    type:'post',
	    dataType:'html',
	    success:function(data){
            var time = data.substring(1, data.length - 1).split(', ');
            var start = time[0];
            var end = time[1];
            window.location.href = "/?start=" + start + "&end=" + end;
	    }
    });
}