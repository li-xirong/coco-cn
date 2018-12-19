/**
 * server for page login.jsp, deal with the page initial, and form submit process
 */

$(document).ready(function(e) {
	Messenger.options = {
        extraClasses: 'messenger-fixed messenger-on-bottom messenger-on-right',
        theme: 'future'
    }
    logFail();
});

function logFail(){
    state = $("#login").attr("value");
	if(state == 2){
		//no such user
		$("#login-input").parent().parent().attr("class", "form-group has-error");
	}
	else if(state == 3){
		//wrong password
		$("#login-password").parent().parent().attr("class", "form-group has-error");
	}
	else if(state == 4){
         Messenger().post({
            message: '用户session已过期 请重新登陆',
            type: 'error',
            showCloseButton: true
        });
	}
}