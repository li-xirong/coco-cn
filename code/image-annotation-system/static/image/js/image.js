var selected = 0;

jQuery(document).ready(function(){
    selected = 0;
    jQuery("#for-user").val($("#for-user").text());
    rank = $("#mult").attr("value");
    if(rank != 0){
        selected = rank;
        $("#sen" + rank).attr("class", "btn info btn-block add");
    }

    label_init();

    Messenger.options = {
        extraClasses: 'messenger-fixed messenger-on-bottom messenger-on-right',
        theme: 'future'
    }

    judge = $("#judge").attr("value");
    if(judge == "True"){
        Messenger().post({
            message: '已经是最后一张图片。',
            type: 'success',
            showCloseButton: true
        });
    }

    $('#image').load(function() {
         var maxWidth = $(document.body).width() * (5 / 12);
         var maxHeight = $("#judge").height() * 0.8;
         var ratio = 0;
         var width = $(this).width();
         var height = $(this).height();

        if(height > maxHeight && width > maxWidth){
            var ratioh = maxHeight / height;
            var ratiow = maxWidth / width;
            ratio = ratioh > ratiow ? ratiow : ratioh;
            $(this).css("height", height * ratio);
            $(this).css("width", width * ratio);
        }
        else if(height > maxHeight){
             ratio = maxHeight / height;
             $(this).css("height", maxHeight);
             width = width * ratio;
             $(this).css("width", width * ratio);
        }else if(width > maxWidth){
             ratio = maxWidth / width;
             $(this).css("width", maxWidth);
             height = height * ratio;
             $(this).css("height", height);
        }
     });
})

function label_init(){
    var labels = $("#label").attr("value").split(', ');
    var s_label = $("#s_label").attr("value");
    for(var i = 0; i < labels.length; i++){
        var btn = $("<button></button>");
        btn.attr("class", "btn btn-primary labels");
        btn.attr("onclick", "addLabels(this)");
        btn.text(labels[i]);
        btn.appendTo($("#label"));
    }
    if(s_label == ""){
        s_labels = [];
    }
    else{
        s_labels = s_label.split(', ');
    }

    var input = $("#tagsinput");
    input.tagsinput({
        typeahead: {
            source: s_labels
        },
        allowDuplicates: false,
        trimValue: true
    });
    input.tagsinput('removeAll');
    for(var i = 0; i < s_labels.length; i++){
        input.tagsinput("add", s_labels[i]);
    }
}

function addLabels(view){
    var text = $(view).text();
    var input = $("#tagsinput");
    labels = input.tagsinput('items')
    if(labels.indexOf(text) == -1){
        input.tagsinput("add", text);
    }
}

function change(view){
    $("#sen" + selected).attr("class", "btn info btn-block");

    id = jQuery(view).attr('id');
    new_selected = id.replace(/[^0-9]/ig, "");

    if(new_selected != selected){
        content = jQuery(view).text();
        jQuery("#for-user").val(jQuery.trim(content));
        $(view).attr("class", "btn info btn-block add");
        selected = new_selected;
    }else{
        selected = 0;
        $("#for-user").val("");
    }

    $(view).css("color", "#000000");
}

function cal_chars(){
	var len = document.getElementById("for-user").value.trim().length;
	document.getElementById("charnum").innerText = len;
}

function checklabels(labarr, sent){
	var flag = 0;
	for(var i = 0; i < labarr.length; i++){
		if(sent.search(labarr[i]) == -1){
			flag = 1;
			return flag;
		}
	}
	return flag;
}

function submit(){
    sentence = jQuery("#for-user").val();
    if(selected == 0){
        selectedSentence = "";
    }
	else{
        selectedSentence = jQuery.trim(jQuery("#sen" + selected).text());
    }
    image_id = $("#image").attr("name");
    var input = $("#tagsinput").tagsinput('items');
    var labels = input.join(', ');
    //var syslabels = $("#label").attr("value");

    /*Messenger().post({
        message: strsent,
        type: 'error',
        showCloseButton: true
    });*/

    if(sentence.trim() == "" | input.length == 0){
        Messenger().post({
            message: '标签与图片描述都不得为空。',
            type: 'error',
            showCloseButton: true
        });

        return;
    }
    else if(input.length < 3){
        Messenger().post({
            message: '标签个数不得少于3个。',
            type: 'error',
            showCloseButton: true
        });

        return;
    }
    else if(sentence.trim().length < 18){
        Messenger().post({
            message: '图片描述的句子长度不得小于18字。',
            type: 'error',
            showCloseButton: true
        });

        return;
    }
	else if(checklabels(input, sentence.trim()) == 0){
		Messenger().post({
			message: '输入标签不能完全被输入句子包含。',
			type: 'error',
			showCloseButton: true
		});

		return;
	}

    edit = $("#submit").attr("value")
    $.ajax({
	    url:"image",
	    data:{
	        image_id : image_id,
	    	sentence : sentence,
	    	selected : selected,
	    	selectedSentence : selectedSentence,
	    	label : labels,
	    	edit : edit
	    },
	    type:'post',
	    dataType:'html',
	    success:function(data){
            window.location.href = "/image";
	    }
    });
}
