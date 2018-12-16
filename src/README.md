# Music Generation on the Server Side

In order to run the server, you need `sudo` access. You don't want to do 
this directly through `sudo python` because for some reason that screws
up with the `anaconda` python pathing. So you instead want to do:

```
> sudo -s # switch into sudo user mode

> python server.py
```
