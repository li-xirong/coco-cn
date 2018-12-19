;(function($) {
    var Page = function(ele, pageNum, currentPage, currentPoint, opt) {
        this.$element = ele,
        this.$pageNum = parseInt(pageNum),
        this.$currentPage = parseInt(currentPage),
        this.$currentPoint = parseInt(currentPoint),
        this.defaults = {
            'maxPageNum': 10
        },
        this.options = $.extend({}, this.defaults, opt),
        this.$point = currentPoint == undefined ? Math.floor(this.$currentPage / this.options.maxPageNum) : currentPoint
    };

    Page.prototype = {
        init: function() {
            var hrefSearch = window.location.search;
            var hrefIndex = hrefSearch.indexOf("page=");
            var targetHref = "";
            if(hrefIndex == -1){
                if(hrefSearch.indexOf("?") == -1){
                    targetHref = "?page=";
                } else {
                    targetHref = hrefSearch + "&page=";
                }
            } else {
                var index = hrefSearch.substring(hrefIndex, hrefSearch.length).indexOf("&");
                targetHref = hrefSearch.substring(0, hrefIndex) + hrefSearch.substring(hrefIndex, hrefSearch.length).substring(index == -1 ? hrefSearch.length : index + 1, hrefSearch.length) + (index == -1? "page=" : "&page=");
            }
            if(this.$pageNum == 1 | this.$pageNum <= 0){
                return
            }
            var page_number = this.$pageNum > this.options.maxPageNum ? this.options.maxPageNum : this.$pageNum;
            var div = $("<div style-'margin:0 auto'></div>");
            div.attr("class", "pagination pagination-success");
            var forward = $("<a href='" + targetHref + (this.$currentPage - 1) + "' class='btn btn-success previous'>Previous</a>");
            if(this.$currentPage == 1){
                forward.attr("class", "btn btn-success previous disabled");
            }
            div.append(forward);
            var ul = $("<ul></ul>");

            for(var i = 0; i < page_number && (this.$point * this.options.maxPageNum + (i + 1)) <= this.$pageNum; i++){
                var li = $("<li></li>");
                var a = $("<a>" + (this.$point * this.options.maxPageNum + (i + 1)) + "</a>");
                li.append(a);
                if(this.$point * this.options.maxPageNum + (i + 1) == this.$currentPage){
                    li.attr("class", "active disabled");
                }
                else{
                    a.attr("href", targetHref + (this.$point * this.options.maxPageNum + (i + 1)));
                }
                ul.append(li);
            }
            div.append(ul);
            if(this.$pageNum > this.options.maxPageNum){
                var li = $("<li class='pagination-dropdown dropup active'></li>");
                ul.append(li);
                var a = $("<a href='#fakelink' class='dropdown-toggle' data-toggle='dropdown'></a>");
                li.append(a);
                var i = $("<i class='fui-triangle-up'></i>");
                a.append(i);
                var dropdown = $("<ul class='dropdown-menu'></ul>");
                li.append(dropdown);
                groupPage = function(id, pageNum, currentPage, currentPoint){
                    $("#" + id).empty();
                    $("#" + id).myPlugin(pageNum, currentPage, currentPoint);
                };
                var i;
                for(i = 0; i < Math.ceil(this.$pageNum / this.options.maxPageNum) - 1; i++){
                    var dropdown_li = $("<li><a>" + (1 + i * this.options.maxPageNum) + "–" + ((i + 1) * this.options.maxPageNum) + "</a></li>");

                    dropdown_li.attr("onclick", "groupPage('" + this.$element.attr("id") + "', " + this.$pageNum + ", " + this.$currentPage + ", " + i + ")")
                    if(i == this.$point){
                        dropdown_li.attr("class", "active");
                    }
                    dropdown.append(dropdown_li);
                }
                var dropdown_li = $("<li><a>" + (1 + i * this.options.maxPageNum) + "–" + (this.$pageNum) + "</a></li>");
                dropdown.append(dropdown_li);
                dropdown_li.attr("onclick", "groupPage('" + this.$element.attr("id") + "', " + this.$pageNum + ", " + this.$currentPage + ", " + i + ")")
                if(i == this.$point){
                    dropdown_li.attr("class", "active");
                }
            }
            var back = parseInt(this.$currentPage) + 1;
            var backward = $("<a href='" + targetHref + back + "' class='btn btn-success next'>Next</a>");
            if(this.$currentPage == this.$pageNum){
                backward.attr("class", "btn btn-success next disabled");
            }
            div.append(backward);
            return this.$element.append(div);
        }
    };
    //在插件中使用Beautifier对象
    $.fn.myPlugin = function(pageNum, currentPage, currentPoint, options) {
        //创建Beautifier的实体
        var page = new Page(this, pageNum, currentPage, currentPoint, options);
        //调用其方法
        return page.init();
    }
})(jQuery);