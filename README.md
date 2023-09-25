# Tiny_StyleTransfer
Very tiny styletransfer networks with only thousands of parameters, sutiable for embedded devices.

Only 3579 parameters. 220Mflops and several millseconds are needed when input image size = 256*256. Can be much more faster by quantize to int8.

Its performence is similiar to full model with over 1M parameters.(Visually)

只有3579个参数.在256*256的图片上风格迁移时,只需要0.22Gflops计算量和毫秒级别的耗时,并且可以通过量化为int8变的更快.

效果看上去和具有1000000+参数的完整网络差的不多.
