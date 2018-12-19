jQuery(document).ready(function(){
    Messenger.options = {
        extraClasses: 'messenger-fixed messenger-on-bottom messenger-on-right',
        theme: 'future'
    }
    judge = $("#show").attr("value");
    if(judge == 0){
        Messenger().post({
            message: '没有足够的内容了！',
            type: 'error',
            showCloseButton: true
        });
    }
    else{
        var page = $("#page");
        var pageNum = page.attr("value");
        var current = page.attr("name");
        page.myPlugin(pageNum, current);
    }
});

function image(id){
    window.location.href = "/image?image_id=" + id;
}