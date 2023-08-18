console.log("home js")



var class_on = undefined;

console.log($("video").length)


function mover(d){



    let class_ = $(d).attr("class").split(" ")[1]

    if(class_ != class_on){

        if(class_on){
            console.log(class_on)
            $(`#${class_on}`).get(0).pause();
            $(`#${class_on}`).css("display", "none")
            
        }
        console.log(class_)
    
        
        $(`#${class_}`).css("display", "inline")
        $(`#${class_}`).get(0).play();
        
    
        class_on = class_


    }
    
}


// function mout(d){
//     console.log("mouseover")

//     // $(".home-img-container").css("display", "none")
//     // $("#bayes_inference_img").attr("display", "block")

// }