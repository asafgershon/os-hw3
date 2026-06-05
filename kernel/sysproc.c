#include "types.h"
#include "riscv.h"
#include "defs.h"
#include "param.h"
#include "memlayout.h"
#include "spinlock.h"
#include "proc.h"

uint64
sys_exit(void)
{
  int n;
  argint(0, &n);
  exit(n);
  return 0;  // not reached
}

uint64
sys_getpid(void)
{
  return myproc()->pid;
}

uint64
sys_fork(void)
{
  return fork();
}

uint64
sys_wait(void)
{
  uint64 p;
  argaddr(0, &p);
  return wait(p);
}

uint64
sys_sbrk(void)
{
  uint64 addr;
  int n;

  argint(0, &n);
  addr = myproc()->sz;
  if(growproc(n) < 0)
    return -1;
  return addr;
}

uint64
sys_sleep(void)
{
  int n;
  uint ticks0;

  argint(0, &n);
  acquire(&tickslock);
  ticks0 = ticks;
  while(ticks - ticks0 < n){
    if(killed(myproc())){
      release(&tickslock);
      return -1;
    }
    sleep(&ticks, &tickslock);
  }
  release(&tickslock);
  return 0;
}

uint64
sys_kill(void)
{
  int pid;

  argint(0, &pid);
  return kill(pid);
}

// return how many clock tick interrupts have occurred
// since start.
uint64
sys_uptime(void)
{
  uint xticks;

  acquire(&tickslock);
  xticks = ticks;
  release(&tickslock);
  return xticks;
}

// sys_flip_display: zero-copy page flip.
//
// Syscall argument 0: user virtual address of a page-aligned buffer
// that is exactly GPU_FB_PAGES (300) * PGSIZE bytes (i.e. 640x480x4 =
// 1,228,800 bytes).  The buffer must already be fully mapped in the
// calling process's address space.
uint64
sys_flip_display(void)
{
  uint64 buf;
  struct proc *p = myproc();

  argaddr(0, &buf);

  if (buf % PGSIZE != 0)
    return -1;

  // Policy check: every page of the buffer must be mapped with user permission.
  // This validation belongs here (syscall layer) rather than in the GPU driver.
  for (int i = 0; i < GPU_FB_PAGES; i++) {
    pte_t *pte = walk(p->pagetable, buf + (uint64)i * PGSIZE, 0);
    if (pte == 0 || (*pte & (PTE_V | PTE_U)) != (PTE_V | PTE_U))
      return -1;
  }

  if (virtio_gpu_flip(p->pagetable, buf) < 0)
    return -1;

  p->fb_flip_active = 1;
  p->fb_flip_va = buf;  // remember which buffer is currently displayed
  return 0;
}

// sys_map_display: map the GPU's kernel framebuffer pages (fb[]) directly
// into the calling process's address space with PTE_U|PTE_R|PTE_W.
//
// Syscall argument 0: desired user virtual address (must be page-aligned).
//   Pass 0 to let the kernel auto-select the next available VA above p->sz.
//
// Returns the mapped virtual address on success, (uint64)-1 on failure.
uint64
sys_map_display(void)
{
  uint64 addr;
  struct proc *p = myproc();

  argaddr(0, &addr);

  // Reject a second mapping: the first one would become orphaned because
  // freeproc/exec only tracks a single fb_map_va.  The caller must call
  // unmap_display() before mapping again.
  if (p->fb_map_va != 0)
    return -1;

  // Track whether the caller asked for auto-selection so we know
  // whether to advance p->sz after the mapping is installed.
  int auto_select = (addr == 0);

  if (auto_select) {
    // Dynamically pick the first free page-aligned VA above the current
    // heap boundary, exactly as the spec requires ("above p->sz").
    addr = PGROUNDUP(p->sz);
  } else {
    if (addr % PGSIZE != 0)
      return -1;
  }

  // Region must fit below the trapframe.
  if (addr + (uint64)GPU_FB_PAGES * PGSIZE > TRAPFRAME)
    return -1;

  // Check every page in the target range is currently unmapped.
  for (int i = 0; i < GPU_FB_PAGES; i++) {
    pte_t *pte = walk(p->pagetable, addr + (uint64)i * PGSIZE, 0);
    if (pte != 0 && (*pte & PTE_V) != 0)
      return -1;
  }

  // Install one PTE per framebuffer page (do_free=0 on unmap: kernel owns them).
  for (int i = 0; i < GPU_FB_PAGES; i++) {
    uint64 pa = virtio_gpu_fb_pa(i);
    if (mappages(p->pagetable, addr + (uint64)i * PGSIZE,
                 PGSIZE, pa, PTE_R | PTE_W | PTE_U) < 0) {
      if (i > 0)
        uvmunmap(p->pagetable, addr, i, 0);
      return -1;
    }
  }

  p->fb_map_va = addr;

  // For the auto-select case, advance p->sz past the new mapping so that
  // a subsequent sbrk() cannot grow the heap into the framebuffer region.
  if (auto_select)
    p->sz = addr + (uint64)GPU_FB_PAGES * PGSIZE;

  return addr;
}

// sys_unmap_display: remove the GPU framebuffer mapping that was installed
// by a previous map_display() call.  Returns 0 on success, -1 if no
// mapping is active.
uint64
sys_unmap_display(void)
{
  struct proc *p = myproc();

  if (p->fb_map_va == 0)
    return -1;

  uvmunmap(p->pagetable, p->fb_map_va, GPU_FB_PAGES, 0);
  p->fb_map_va = 0;
  return 0;
}
