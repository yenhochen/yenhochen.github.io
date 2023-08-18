// create data. Resolution of the data
var nx = 5;
var ny = 5;
var tilewidth = 7;
var tileheight = 7;
var paddingx = 3;
var paddingy =3;
var kernel = 3;
var stride = 1;
var padding = 0;
var outpad = 0;
var dilation = 1;


var offx = 10;
var offy = 70;

var matrix_pad = 74; 

var nx_out, ny_out = compute_outsize(nx, ny, kernel, stride, padding, dilation, outpad)
var wx, wy, ox, oy, nx_int, ny_int, otx, oty = get_offset(nx, ny, kernel, stride, padding, offx, offy, nx_out, ny_out, dilation, outpad)

let width = wx.reduce((a, b) => a + b, 0)+matrix_pad*4
const svg = d3.select(".d3post1")
              .append("svg")
              .attr("width", width)
              .attr("height", 400);

var text = ["★★", "=","★", "=", "Transposed Conv.", "Conv. Formulation", "Output", "5x5", "9x9", "7x7"]

// d3.select()



var img_color = "#b23a48"
var kernel_color = "#F1A253"
var int_color = "#4A6793"
var out_color = "#39A482"
var gray_color = "#B9B9B9"
var scale = chroma.scale(['124559',"62b6cb", "9bf6ff"]).mode("hsl")

data = makefullData(nx, ny, offx, offy, kernel, stride, padding, dilation, outpad)

// draw on svg
updateText(otx)
updateRectangles(data)


d3.select("#slider-k")
    .on("input", d=>{
     kernel = d3.select("#slider-k").node().value
     new_data = makefullData(nx, ny, offx, offy, kernel, stride, padding, dilation, outpad)
     updateRectangles(new_data);
 })

d3.select("#slider-k")
   .on("input", d=>{
    kernel = d3.select("#slider-k").node().value
    new_data = makefullData(nx, ny, offx, offy, kernel, stride, padding, dilation, outpad)
    updateRectangles(new_data);
})

d3.select("#slider-w")
   .on("input", d=>{
    nx = d3.select("#slider-w").node().value
    new_data = makefullData(nx, ny, offx, offy, kernel, stride, padding, dilation, outpad)
    updateRectangles(new_data);
})

d3.select("#slider-h")
   .on("input", d=>{
    ny = d3.select("#slider-h").node().value
    new_data = makefullData(nx, ny, offx, offy, kernel, stride, padding, dilation, outpad)
    updateRectangles(new_data);
})

d3.select("#slider-s")
   .on("input", d=>{
    stride = d3.select("#slider-s").node().value
    new_data = makefullData(nx, ny, offx, offy, kernel, stride, padding, dilation, outpad)
    updateRectangles(new_data);
})

d3.select("#slider-p")
   .on("input", d=>{
    padding = d3.select("#slider-p").node().value
    new_data = makefullData(nx, ny, offx, offy, kernel, stride, padding, dilation, outpad)
    updateRectangles(new_data);
})


d3.select("#slider-d")
   .on("input", d=>{
    dilation = d3.select("#slider-d").node().value
    new_data = makefullData(nx, ny, offx, offy, kernel, stride, padding, dilation, outpad)
    updateRectangles(new_data);
})


d3.select("#slider-op")
   .on("input", d=>{
    outpad = d3.select("#slider-op").node().value
    
    new_data = makefullData(nx, ny, offx, offy, kernel, stride, padding, dilation, outpad)
    updateRectangles(new_data);
})

function createcolorsf(n_colors, f_color){
    let colors = []
    for(let i=0; i< n_colors; i++){
        colors.push({"color":f_color(i/n_colors).hex()})
    }
    return colors
}


function createcolors(n_colors, color){
    let colors = []
    for(let i=0; i< n_colors; i++){
        colors.push({"color":color})
    }
    return colors
}



function makefullData(nx, ny, offx, offy, kernel, stride, padding, dilation, outpad){

    nx = parseInt(nx)
    ny = parseInt(ny)
    kernel = parseInt(kernel)
    stride = parseInt(stride)
    padding = parseInt(padding)
    dilation = parseInt(dilation)
    outpad = parseInt(outpad)


    // console.log("MAKE FULL DATA", nx, ny, offx, offy, kernel, stride, padding, dilation, outpad)
    // console.log("TYPEOF", typeof nx)
    nx_out, ny_out = compute_outsize(nx, ny, kernel, stride, padding, dilation, outpad)
    wx, wy, ox, oy, nx_int, ny_int, otx, oty = get_offset(parseInt(nx), parseInt(ny), kernel, stride, padding, offx, offy, nx_out, ny_out, dilation, outpad)
    nx_int = parseInt(nx_int)
    ny_int = parseInt(ny_int)

    console.log("INTERMETDIATE", nx_int, ny_int)

    // console.log(wx, wy, ox, oy, nx_int, ny_int, otx, oty, "asdfg4r")


    let width = wx.reduce((a, b) => a + b, 0)+matrix_pad*4+50
    svg.attr("width", width)


    updateText(otx);
    

    data_img = createData(nx, ny, ox[0], oy[0], 0)
    data_kernel1 = createData(kernel, kernel, ox[1], oy[1], 1)
    data_int = createData(nx_int, ny_int, ox[2], oy[2], 2)
    data_kernel2 = createData(kernel, kernel, ox[3], oy[3],3)
    data_out = createData(nx_out, ny_out, ox[4], oy[4],4)
    data = [...data_img, ...data_kernel1, ...data_int, ...data_kernel2, ...data_out]

    img_colors = createcolors(data_img.length, img_color)
    // k1_colors = createcolors(data_kernel1.length, kernel_color)
    int_colors = createcolors(data_int.length, img_color)
    out_colors = createcolors(data_out.length, out_color)

    k1_colors = createcolorsf(data_kernel1.length, scale)
    k2_colors = [...k1_colors].reverse()

    var z = stride-1;
    var p_prime = dilation*(kernel-1)-padding+outpad;
    
    console.log("nxny", nx, ny)
    int_colors.forEach((e,i)=>{
        // gray out side padding
        if(data_int[i].ix < p_prime || data_int[i].ix >= nx+p_prime+z*(nx-1)){
            int_colors[i].color = gray_color
        }
        if(data_int[i].iy < p_prime || data_int[i].iy >= ny+p_prime+z*(ny-1)){
            int_colors[i].color = gray_color
        }

         // gray out internal padding        
         if(data_int[i].ix > p_prime && (data_int[i].ix-p_prime)%stride != 0 ){
            int_colors[i].color = gray_color
        }
        if(data_int[i].iy > p_prime &&  (data_int[i].iy-p_prime )%stride!=0){
            int_colors[i].color = gray_color
        }
    })

    var color_data = [...img_colors, ...k1_colors, ...int_colors, ...k2_colors, ...out_colors]

    data.forEach((e,i)=>{
        data[i] = {...data[i], ...color_data[i]}
    })

    return data
}


function updateText(otx){


    text_data = []

    for(let i=0; i < otx.length; i++){
    text_data.push({"otx": otx[i],
                    "oty": oty[i],
                    "text": text[i]})
    }



    svg.selectAll("text")
              .data(text_data)
              .join("text")
              .attr("x", d=>{return d.otx})
              .attr("y", d=>{return d.oty})
              .text(d=>{return d.text})
              .attr("text-anchor", "middle")
              .attr("alignment-baseline", "middle")

}

function updateRectangles(data){

    kernel = parseInt(kernel)
    outpad = parseInt(outpad)
    dilation = parseInt(dilation)

    svg.selectAll("rect")
        .data(data)
        .join("rect")
        .attr("x", d=>{return d.x})
        .attr("y", d=>{return d.y})
        .attr("ix", d=>{return d.ix})
        .attr("iy", d=>{return d.iy})
        .attr("m", d=>{return d.m})
        .attr('stroke', 'black')
        .attr("stroke-width", 1)
        .attr('fill', d=>{return d.color})
        .attr('width', tilewidth)
        .attr('height', tileheight)
        .on("mouseover",function(e,d){
            // console.log(e, d)
            if (d3.select(this).attr("m") == 4){
                d3.select(this)
                  .attr("stroke-width", 2.5)

                let out_x = parseInt(d3.select(this).attr("ix"))
                let out_y = parseInt(d3.select(this).attr("iy"))
                console.log(out_x)

                console.log("sdfvb", kernel)

                for(let i=0; i<dilation*kernel; i=i+dilation){
                    for(let j=0; j<dilation*kernel; j=j+dilation){
                        let intx = out_x+i+outpad;
                        let inty = out_y+j+outpad;
                        let r = d3.select(`rect[m='2'][ix='${intx}'][iy='${inty}']`)
                          .attr("stroke-width", 2.5)
                        // console.log(`rect[m='2'][ix='${intx}'][iy='${inty}']`)
                        // console.log(r)
                    }
                }
                
                

            }


        })
        .on("mouseout",function(e,d){
            // console.log(e, d)
            d3.select(this)
              .attr("stroke-width", 1)

              let out_x = parseInt(d3.select(this).attr("ix"))
                let out_y = parseInt(d3.select(this).attr("iy"))

                for(let i=0; i<dilation*kernel; i=i+dilation){
                    for(let j=0; j<dilation*kernel; j=j+dilation){
                        let intx = out_x+i+outpad;
                        let inty = out_y+j+outpad;
                        let r = d3.select(`rect[m='2'][ix='${intx}'][iy='${inty}']`)
                          .attr("stroke-width", 1)
                        // console.log(`rect[m='2'][ix='${intx}'][iy='${inty}']`)
                        // console.log(r)
                    }
                }
        })
}


function createData(nx, ny, offsetx, offsety, m){

    data = []
    for(let j = 0; j < ny; j++){
    for (let i = 0; i < nx; i++) {
        
            data.push({
                "x": i*(tilewidth + paddingx)+offsetx,
                "y": j*(tileheight + paddingy)+offsety,
                "ix": i,
                "iy": j,
                "m": m
            })
        }
    }
    return data
}


function get_offset(nx, ny, kernel, stride, padding, offx, offy, nx_out, ny_out, dilation, outpad){

    var z = stride-1;
    var p_prime = dilation*(kernel-1)-padding +outpad;
    // console.log("nx", nx, ny, p_prime, z) # ok
    ny_int = ((ny-1)*z+ny)+p_prime*2;
    nx_int = ((nx-1)*z+nx)+p_prime*2;
    // console.log("intermediate_size", nx_int, ny_int)
    // var nx_int = ((ny-1)*z+ny);
    // var ny_int = ((nx-1)*z+nx);

    var wx1 = nx*(tilewidth+paddingx)
    var wx2 = kernel*(tilewidth+paddingx)
    var wx3 = nx_int*(tilewidth+paddingx)
    var wx4 = wx2
    var wx5 = nx_out*(tilewidth+paddingx)

    wx = [wx1, wx2, wx3, wx4, wx5]


    var wy1 = ny*(tileheight+paddingy)
    var wy2 = kernel*(tileheight+paddingy)
    var wy3 = ny_int*(tileheight+paddingy)
    var wy4 = wy2
    var wy5 = ny_out*(tileheight+paddingy)

    wy = [wy1, wy2, wy3, wy4, wy5]

    var wx_max = Math.max(...wx)
    var wy_max = Math.max(...wy)
    var maxy_i = wy.indexOf(Math.max(...wy));


    var ox1 = offx;
    var ox2 = ox1 + wx1 + matrix_pad;
    var ox3 = ox2 + wx2 + matrix_pad;
    var ox4 = ox3 + wx3 + matrix_pad;
    var ox5 = ox4 + wx4 + matrix_pad;

    ox = [ox1, ox2, ox3, ox4, ox5]

    var oy1 = offy+wy_max/2-wy1/2;
    var oy2 = offy+wy_max/2-wy2/2;
    var oy3 = offy+wy_max/2-wy3/2;
    var oy4 = offy+wy_max/2-wy4/2;
    var oy5 = offy+wy_max/2-wy5/2;


    var otx1 = offx+wx1 + matrix_pad/2
    var otx2 = ox2 + wx2 + matrix_pad/2
    var otx3 = ox3 + wx3 + matrix_pad/2
    var otx4 = ox4 + wx4 + matrix_pad/2
    var otx5 = otx1+10
    var otx6 = otx3
    var otx7 = ox4 + wx4 + matrix_pad + wx5/2

    var oty1 = offy + wy_max/2
    var oty2 = offy + wy_max/2
    var oty3 = offy + wy_max/2
    var oty4 = offy + wy_max/2
    var oty5 = 10
    var oty6 = 10
    var oty7 = 10

    oy = [oy1, oy2, oy3, oy4, oy5]
    otx = [otx1, otx2, otx3, otx4, otx5, otx6, otx7]
    oty = [oty1, oty2, oty3, oty4, oty5, oty6, oty7]

    return wx, wy, ox, oy, nx_int, ny_int, otx, oty
}

function compute_outsize(nx, ny, kernel, stride, padding, dilation, outpad){
    nx = parseInt(nx)
    ny = parseInt(ny)
    kernel = parseInt(kernel)
    stride = parseInt(stride)
    padding = parseInt(padding)
    dilation = parseInt(dilation)
    outpad = parseInt(outpad)

    nx_out = (nx-1)*stride-2*padding+dilation*(kernel-1) + outpad + 1
    ny_out = (ny-1)*stride-2*padding+dilation*(kernel-1) + outpad + 1
    return nx_out, ny_out
}
