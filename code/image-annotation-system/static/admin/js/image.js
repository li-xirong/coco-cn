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
            $(this).css("left", (maxWidth - width * ratio) / 2);
        }
        else if(height > maxHeight){
            ratio = maxHeight / height;
            $(this).css("height", maxHeight);
            width = width * ratio;
            $(this).css("width", width * ratio);
            $(this).css("left", (maxWidth - width * ratio) / 2);
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