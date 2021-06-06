var loadFile = function(event) {
	var image = document.getElementById('output');
	image.src = URL.createObjectURL(event.target.files[0]);
};

function videoToggle() {
	console.log('clicked')
	if (document.getElementById("video_rep").src == "static/images/video_replacement.png") {
		link = "{{"+" url_for('"+"video_feed'"+") }}";
		document.getElementById("video_rep").src = link;
		}
	else {
		document.getElementById("video_rep").src = "static/images/video_replacement.png";
		}
	}